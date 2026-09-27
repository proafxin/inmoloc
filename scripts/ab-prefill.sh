#!/usr/bin/env bash
# Prompt processing of page images, vanilla llama.cpp (the worktree at the fork's base commit) against this fork.
# The OCR pool of citadel's run_ocr_pool.py (PDF pages, images, embedded pictures from ~/Downloads/ocr_input) is sent with
# max_tokens 1, so the time is prompt processing (image encoding included) and not generation. Same requests at once on
# both servers, alternating V F V F with a fresh server per run, since single runs vary by ~10% on this card.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
VAN=${VAN:-/home/masterkenway/Projects/llama.cpp-vanilla}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/ab_prefill}
MODELS=$CITADEL/data/gguf_models
IMG=nvidia/cuda:13.3.0-devel-ubuntu24.04
ROUNDS=${ROUNDS:-2}
CONC=${CONC:-8}            # requests at once, and --parallel of both servers
MAX_PAGES=${MAX_PAGES:-3}  # pages per PDF

COMMON="--model /models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf} --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k ${KV:-q8_0} --cache-type-v ${KV:-q8_0} --flash-attn on --alias lm
        --ctx-size ${CTX:-131072} --parallel $CONC --cache-ram 0 -lv 4"

mkdir -p $OUT

build() { # $1 = source tree
    docker run --rm -i --gpus all -v $1:/src -w /src $IMG bash -c "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && git config --global --add safe.directory '*' && [ -f build/CMakeCache.txt ] || cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build --config Release -j --target llama-server" 2>&1 | tail -1
}

run() { # $1 = vanilla|fork, $2 = round
    local label=$1-$2 tree=$SRC extra="--rs-rollback replay"
    if [ $1 = vanilla ]; then tree=$VAN; extra=""; fi
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $tree:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server $COMMON $extra >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$label.log 2>&1
        echo "$label did not start, see $OUT/server-$label.log"
        return
    fi
    rm -rf $OUT/pool-$label
    ( cd $CITADEL && uv run python data/model_baselines/run_ocr_pool.py $OUT/pool-$label $CONC $MAX_PAGES 1 > $OUT/pool-$label.log 2>&1 )
    echo "$label: $(tail -1 $OUT/pool-$label.log)"
    curl -s localhost:8100/metrics | grep -v '^#' > $OUT/metrics-$label.txt
    docker logs lm > $OUT/server-$label.log 2>&1
    docker rm -f lm >/dev/null 2>&1
}

echo "building the fork"; build $SRC
echo "building vanilla";  build $VAN

for r in $(seq 1 $ROUNDS); do
    run vanilla $r
    run fork    $r
done

# where the time goes, from the server logs: the image embeddings decoded through the model (logged by the mtmd helper in
# both servers), and on the fork the media time of the server loop and the encoder/decoder metrics
for f in $OUT/server-*.log; do
    label=$(basename $f .log | sed 's/^server-//')
    dec=$(grep -h "image decoded" $f | sed -E 's/.* in ([0-9]+) ms.*/\1/' | awk '{ s += $1; n++ } END { printf "%d image batches decoded in %.1f s", n, s/1000 }')
    media=$(grep -h "slow iteration" $f | sed -E 's/.*media ([0-9]+) ms \/ ([0-9]+) tokens.*/\1 \2/' | awk '{ m += $1; t += $2 } END { if (NR) printf ", loop media %.1f s for %d tokens", m/1000, t }')
    mt=$(grep -h -E "mtmd_(encode|decode)_seconds_total" $OUT/metrics-$label.txt 2>/dev/null | awk '{ printf ", %s %.1f s", $1, $2 }' | sed 's/llamacpp://g')
    echo "$label: $dec$media$mt"
done

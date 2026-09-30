#!/usr/bin/env bash
# The committed code (A) against the uncommitted changes of the working tree (B), alternating A B A B, each run on a
# freshly started server so that no run finds the documents of another in the prompt cache; the workload is identical.
# The changes are set aside with git stash for the A runs and restored afterwards, also when a run fails.
set -u

SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/home/masterkenway/Projects/citadel/data/model_baselines/ab_local}
MODELS=${MODELS:-/home/masterkenway/Projects/citadel/data/gguf_models}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}
ROUNDS=${ROUNDS:-2}
WORKLOAD=${WORKLOAD:-bench}   # bench = long documents next to short prompts (BENCH_ARGS), ocr = the citadel OCR battery
BUDGET=${BUDGET:-22G}         # the device memory the server may use (the GPU also drives the desktop)
CTX=${CTX:-131072}           # tokens of the KV cache shared by all requests

mkdir -p $OUT
cd $SRC

git diff --quiet -- src ggml tools common include && { echo "no uncommitted code changes to compare"; exit 1; }

restore() { git stash list | grep -q ab-local-change && git stash pop -q; }
trap restore EXIT

# a failed build must stop the comparison: the run after it would use the binary of the other side
build() {
    local res
    res=$(docker run --rm -i --gpus all -v $SRC:/src -w /src $IMG bash -c "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && git config --global --add safe.directory /src && cmake --build build --config Release -j --target llama-server" 2>&1 | tail -1)
    echo "$res"
    case "$res" in *"Built target llama-server"*) ;; *) echo "build failed, stopping"; exit 1 ;; esac
}

run() { # $1 = label
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
        --model /models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf} --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024 \
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100 \
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k ${KV:-q8_0} --cache-type-v ${KV:-q8_0} --flash-attn on \
        --alias lm --ctx-size $CTX --parallel 32 --rs-rollback replay --cache-ram 4096 -lv 4 \
        ${BUDGET:+--vram-budget $BUDGET} ${WORKLOAD:+$([ $WORKLOAD = ocr ] && echo "--mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024")} >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    echo "--- $1"
    docker logs lm 2>&1 | grep -h common_budget_params | tail -1 | cut -c1-200
    if [ $WORKLOAD = ocr ]; then
        rm -rf $OUT/ocr-$1; mkdir -p $OUT/ocr-$1
        local t0=$(date +%s.%N)
        ( cd ${CITADEL:-/home/masterkenway/Projects/citadel} && uv run python data/model_baselines/run_ocr_full.py \
            $OUT/ocr-$1/ocr $OUT/ocr-$1/ocr_images > $OUT/ocr-$1/run.log 2>&1 )
        echo "OCR battery wall: $(echo "$(date +%s.%N) - $t0" | bc) s, $(ls $OUT/ocr-$1/ocr 2>/dev/null | wc -l) documents"
        curl -s localhost:8100/metrics | grep -vE '^#' | grep -E 'mtmd_(encode|decode)_seconds_total'
    else
        python3 $SRC/scripts/bench-server-concurrency.py ${BENCH_ARGS:---levels 16 --long 2 --long-tokens 20000 --max-tokens 256} | tail -1
    fi
    docker logs lm > $OUT/server-$1.log 2>&1
    docker rm -f lm >/dev/null 2>&1
    if [ $WORKLOAD = ocr ]; then
        python3 $SRC/scripts/ocr-score.py $OUT/ocr-$1/ocr
    fi
    # iterations of the server loop slow enough to be logged while slots were generating: the stalls of generation
    grep -h "slow iteration" $OUT/server-$1.log | grep -v "generating = 0" \
        | sed -E 's/.*slow iteration: ([0-9]+) ms.*media ([0-9]+) ms.*/\1 \2/' \
        | awk '{ n++; t += $1; m += $2; if ($1 > mx) mx = $1 } END { printf "generation stalls: %d slow iterations, %.1f s in total (%.1f s of it media), longest %d ms\n", n, t/1000, m/1000, mx }'
}

for r in $(seq 1 $ROUNDS); do
    git stash push -q -m ab-local-change -- $(git diff --name-only -- src ggml tools common include)
    build
    run A$r
    git stash pop -q
    build
    run B$r
done

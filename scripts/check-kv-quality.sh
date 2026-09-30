#!/usr/bin/env bash
# Long-context quality of a KV cache type against f16 (scripts/kv-quality.py): the production server with one sequence,
# once per cache type in RUNS, the first being the reference; a second run of the reference type gives the noise floor
# of the GPU (e.g. RUNS="f16 q8_0 f16"). Without the MTP draft: it does not change the greedy output, which the model
# verifies, and an f16 cache leaves no room for it.
set -u

SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/kv_quality}
MODELS=$CITADEL/data/gguf_models
IMG=nvidia/cuda:13.3.0-devel-ubuntu24.04
MODEL=${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf}
CTX=${CTX:-114688}       # the 110k-token document and the question; an f16 cache of 131072 tokens does not fit the budget
BUDGET=${BUDGET:-22G}
RUNS=${RUNS:-"f16 q8_0 f16"}

mkdir -p $OUT

files=()
i=0
for kv in $RUNS; do
    i=$((i + 1))
    label=$kv-$i

    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
        --model /models/$MODEL --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100 \
        --cache-type-k $kv --cache-type-v $kv --flash-attn on --alias lm \
        --ctx-size $CTX --parallel 1 --vram-budget $BUDGET --rs-rollback replay --cache-ram 0 -lv 4 >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$label.log 2>&1
        echo "the server with the $kv cache did not start, see $OUT/server-$label.log"
        exit 1
    fi

    python3 $SRC/scripts/kv-quality.py run --label $label --out $OUT/$label.json --src $SRC
    docker logs lm > $OUT/server-$label.log 2>&1
    docker rm -f lm >/dev/null 2>&1
    files+=($OUT/$label.json)
done

python3 $SRC/scripts/kv-quality.py compare "${files[@]}" | tee $OUT/compare.txt

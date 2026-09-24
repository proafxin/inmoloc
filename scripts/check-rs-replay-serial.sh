#!/usr/bin/env bash
# One request at a time on the CPU, snapshot rollback (LLAMA_RS_REPLAY=0) against replay rollback (=1).
# Without concurrent requests the batches have the same shape in both runs, so the CPU output is deterministic
# and any difference is an error of the replay, not rounding noise.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
OUT=${OUT:-/home/masterkenway/Projects/citadel/data/model_baselines/replay_ab/cpu1}
M=${M:-/models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf}}
MODELS=${MODELS:-/home/masterkenway/Projects/citadel/data/gguf_models}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}
N=${N:-64}

# 1 = replay rollback, 0 = snapshot rollback, see --rs-rollback
rb() { [ "$1" = 1 ] && echo replay || echo snapshot; }

mkdir -p $OUT

PROMPTS=(
    "Explain how bread rises, in plain words."
    "Describe a lighthouse on a stormy winter night."
)

for R in 0 1; do
    docker rm -f lm >/dev/null 2>&1
    # --gpus all only provides libcuda to the CUDA build, -dev none keeps the model on the CPU
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 \
        -v $SRC:/src -v $MODELS:/models:ro -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
        --model $M --chat-template-file /models/chat_template.jinja -dev none -ngl 0 \
        --host 0.0.0.0 --port 8100 --ctx-size 4096 --parallel 1 \
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics \
        --cache-type-k f16 --cache-type-v f16 --flash-attn on --alias lm -lv 5 --rs-rollback $(rb $R) >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$R.log 2>&1
        echo "server $R did not start, see $OUT/server-$R.log"
        exit 1
    fi

    : > $OUT/out-$R.txt
    for P in "${PROMPTS[@]}"; do
        curl -s -m 1800 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":$N,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
            | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"]["content"]))' >> $OUT/out-$R.txt \
            || { echo "request failed on server $R"; docker logs lm > $OUT/server-$R.log 2>&1; exit 1; }
    done

    docker logs lm > $OUT/server-$R.log 2>&1
    docker rm -f lm >/dev/null 2>&1
done

echo "replay run: $(grep -c 'RS rollback' $OUT/server-1.log) rollbacks"
if diff $OUT/out-0.txt $OUT/out-1.txt; then
    echo IDENTICAL
else
    echo DIFFERS
fi

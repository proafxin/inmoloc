#!/usr/bin/env bash
# The committed code (A) against the uncommitted changes of the working tree (B), alternating A B A B, each run on a
# freshly started server so that no run finds the documents of another in the prompt cache; the workload is identical.
# The changes are set aside with git stash for the A runs and restored afterwards, also when a run fails.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
OUT=${OUT:-/home/masterkenway/Projects/citadel/data/model_baselines/ab_local}
MODELS=${MODELS:-/home/masterkenway/Projects/citadel/data/gguf_models}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}
ROUNDS=${ROUNDS:-2}

mkdir -p $OUT
cd $SRC

git diff --quiet -- src ggml tools common && { echo "no uncommitted code changes to compare"; exit 1; }

restore() { git stash list | grep -q ab-local-change && git stash pop -q; }
trap restore EXIT

build() {
    docker run --rm -i --gpus all -v $SRC:/src -w /src $IMG bash -c "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && git config --global --add safe.directory /src && cmake --build build --config Release -j --target llama-server" 2>&1 | tail -1
}

run() { # $1 = label
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
        --model /models/Qwen3.8-27B-UD-IQ3_XXS.gguf --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024 \
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100 \
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k f16 --cache-type-v f16 --flash-attn on \
        --alias lm --ctx-size 65536 --parallel 32 --rs-rollback replay --cache-ram 4096 -lv 4 >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    echo "--- $1"
    python3 $SRC/scripts/bench-server-concurrency.py --levels 16 --long 2 --long-tokens 20000 --max-tokens 256 | tail -1
    docker logs lm > $OUT/server-$1.log 2>&1
    docker rm -f lm >/dev/null 2>&1
}

for r in $(seq 1 $ROUNDS); do
    git stash push -q -m ab-local-change -- $(git diff --name-only -- src ggml tools common)
    build
    run A$r
    git stash pop -q
    build
    run B$r
done

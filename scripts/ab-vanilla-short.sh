#!/usr/bin/env bash
# Vanilla llama.cpp against this fork on short prompts at a few concurrency levels, alternating V F V F with a fresh
# server per run, so that a difference is not read from a single run of each.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/ab_short}
MODELS=$CITADEL/data/gguf_models
IMG_FORK=nvidia/cuda:13.3.0-devel-ubuntu24.04
IMG_VANILLA=ghcr.io/ggml-org/llama.cpp:server-cuda
LEVELS=${LEVELS:-"1 4 8"}
ROUNDS=${ROUNDS:-2}

COMMON="--model /models/Qwen3.8-27B-UD-IQ3_XXS.gguf --chat-template-file /models/chat_template.jinja
        -ngl 999 --host 0.0.0.0 --port 8100 --spec-type draft-mtp --spec-draft-n-max 2 --metrics
        --cache-type-k f16 --cache-type-v f16 --flash-attn on --alias lm"

mkdir -p $OUT

serve() { # $1 = vanilla|fork
    docker rm -f lm >/dev/null 2>&1
    if [ $1 = vanilla ]; then
        docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $MODELS:/models:ro $IMG_VANILLA \
            $COMMON --ctx-size 131072 --parallel 4 >/dev/null
    else
        docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
            -e LD_LIBRARY_PATH=/src/build/bin $IMG_FORK /src/build/bin/llama-server \
            $COMMON --ctx-size 65536 --parallel 32 --rs-rollback replay --cache-ram 4096 -lv 4 >/dev/null
    fi
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$1.log 2>&1
        echo "$1 did not start, see $OUT/server-$1.log"
        exit 1
    fi
}

for r in $(seq 1 $ROUNDS); do
    for S in vanilla fork; do
        serve $S
        echo "--- $S round $r"
        python3 $SRC/scripts/bench-server-concurrency.py --levels $LEVELS --max-tokens 256
        docker logs lm > $OUT/server-$S-$r.log 2>&1
        docker rm -f lm >/dev/null 2>&1
    done
done

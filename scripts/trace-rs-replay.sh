#!/usr/bin/env bash
# short CPU run with the rollback trace on, to see what the replay bookkeeping actually does per step
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
OUT=${OUT:-/tmp/rs-replay}
M=${M:-/models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf}}
MODELS=${MODELS:-/home/masterkenway/Projects/citadel/data/gguf_models}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}

# 1 = replay rollback, 0 = snapshot rollback, see --rs-rollback
rb() { [ "$1" = 1 ] && echo replay || echo snapshot; }

mkdir -p $OUT

docker run --rm -i --gpus all -v $SRC:/src -w /src $IMG bash -c \
    "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && \
     git config --global --add safe.directory /src && cmake --build build --config Release -j --target llama-server" \
    2>&1 | tail -2

docker rm -f lm >/dev/null 2>&1
docker run -d --name lm --gpus all -p 8100:8100 \
    -v $SRC:/src -v $MODELS:/models:ro -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
    --model $M --chat-template-file /models/chat_template.jinja -dev none -ngl 0 \
    --host 0.0.0.0 --port 8100 --ctx-size 2048 --parallel 1 \
    --spec-type draft-mtp --spec-draft-n-max 2 --metrics \
    --cache-type-k ${KV:-q8_0} --cache-type-v ${KV:-q8_0} --flash-attn on --alias lm -lv 5 --rs-rollback replay >/dev/null

until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done

curl -s -m 1800 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 12, separated by commas."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

docker logs lm > $OUT/trace.log 2>&1
docker rm -f lm >/dev/null 2>&1
echo "trace saved to $OUT/trace.log"

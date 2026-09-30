#!/usr/bin/env bash
# Compares the two recurrent-state rollbacks of a hybrid model: the snapshot one (LLAMA_RS_REPLAY=0, one state
# per draft position) and the replay one (LLAMA_RS_REPLAY=1, cached scan inputs replayed after a rollback).
#
# The CPU kernel is deterministic and replaying is exact there, so its outputs must be identical: a difference
# means the rollback bookkeeping is wrong. On the GPU the replayed tokens change the chunk boundaries of the
# scan, so the text may differ while still being correct - the counting check is what decides that.
set -u

SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/tmp/rs-replay}
M=${M:-/models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf}}
MODELS=${MODELS:-/home/masterkenway/Projects/citadel/data/gguf_models}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}

# 1 = replay rollback, 0 = snapshot rollback, see --rs-rollback
rb() { [ "$1" = 1 ] && echo replay || echo snapshot; }

mkdir -p $OUT

serve() { # $1 = LLAMA_RS_REPLAY, rest = extra server args
    local replay=$1; shift
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 \
        -v $SRC:/src -v $MODELS:/models:ro -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
        --model $M --chat-template-file /models/chat_template.jinja \
        --host 0.0.0.0 --port 8100 --spec-type draft-mtp --spec-draft-n-max 2 --metrics \
        --cache-type-k ${KV:-q8_0} --cache-type-v ${KV:-q8_0} --flash-attn on --rs-rollback $(rb $replay) --alias lm -lv 5 "$@" >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
}

ask() { # $1 = prompt (json string), $2 = max tokens
    curl -s -m 1800 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":$1}],\"max_tokens\":$2,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1
}

echo "=== building"
docker run --rm -i --gpus all -v $SRC:/src -w /src $IMG bash -c \
    "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && \
     git config --global --add safe.directory /src && cmake --build build --config Release -j --target llama-server" \
    2>&1 | tail -2

# ONLY_SCATTER=1 skips to the last test
if [ -z "${ONLY_SCATTER:-}" ]; then
echo "=== CPU: outputs must be identical, the kernel is deterministic"
echo "    concurrent natural-language requests: frequent rollbacks, idle cells between busy ones, reused slots"
# snapshot mode runs twice: the requests arrive with slightly different timing each run, so the server may batch
# them differently, and different batch shapes round differently. the second snapshot run is the noise floor
for RUN in 0 1 0b; do
    R=${RUN%b}
    echo "--- CPU LLAMA_RS_REPLAY=$R ($RUN, slow, a 27B model on CPU)"
    serve $R -dev none -ngl 0 --ctx-size 4096 --parallel 4
    python3 $SRC/scripts/rs-replay-load.py --url http://localhost:8100 > $OUT/cpu-$RUN.txt 2>&1
    cut -c1-110 $OUT/cpu-$RUN.txt
    docker logs lm > $OUT/cpu-server-$RUN.log 2>&1
    docker rm -f lm >/dev/null 2>&1
done
echo "replay run: $(grep -c 'RS rollback' $OUT/cpu-server-1.log) rollbacks, $(grep -c 'RS range' $OUT/cpu-server-1.log) steps with uncomputed cells in their range"
cmp_runs() { # $1 $2 = runs, prints the number of differing requests
    python3 - "$OUT/cpu-$1.txt" "$OUT/cpu-$2.txt" <<'PY'
import sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
print(sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)))
PY
}
echo "requests that differ: snapshot vs snapshot $(cmp_runs 0 0b), snapshot vs replay $(cmp_runs 0 1)"
if diff -q $OUT/cpu-0.txt $OUT/cpu-1.txt >/dev/null; then
    echo "CPU IDENTICAL"
else
    echo "CPU DIFFERS"
    diff $OUT/cpu-0.txt $OUT/cpu-1.txt | head -8 | cut -c1-160
fi

# ONLY_CPU=1 stops after the exact comparison
if [ -n "${ONLY_CPU:-}" ]; then exit 0; fi

echo "=== GPU: counting must stay correct, speed and memory are informational"
for R in 0 1; do
    echo "--- LLAMA_RS_REPLAY=$R"
    serve $R -ngl 999 --ctx-size 32768 --parallel 4
    docker logs lm 2>&1 | grep -E 'llama_memory_recurrent: size'
    python3 $SRC/scripts/check-server-preemption.py --url http://localhost:8100 --n-requests 4 --n-predict 800
    echo "(second round, reusing the slots)"
    python3 $SRC/scripts/check-server-preemption.py --url http://localhost:8100 --n-requests 4 --n-predict 800
    python3 $SRC/scripts/bench-server-concurrency.py --url http://localhost:8100 --levels 1 4 --max-tokens 256
    curl -s localhost:8100/metrics | grep -vE '^#' | grep -E 'spec_decode_num_(draft|accepted)_tokens_total'
    docker logs lm > $OUT/gpu-server-$R.log 2>&1
    docker rm -f lm >/dev/null 2>&1
done

fi

echo "=== GPU, requests of different lengths: the short ones finish while the long ones go on, and their idle"
echo "    cells stay in memory in between, so a step's range holds cells it does not compute"
for R in 0 1; do
    echo "--- LLAMA_RS_REPLAY=$R"
    serve $R -ngl 999 --ctx-size 32768 --parallel 6
    for round in 1 2; do
        echo "(round $round)"
        python3 $SRC/scripts/check-server-preemption.py --url http://localhost:8100 --n-requests 3 --n-predict 900 > $OUT/long-$R-$round.txt 2>&1 &
        sleep 2
        python3 $SRC/scripts/check-server-preemption.py --url http://localhost:8100 --n-requests 3 --n-predict 120 > $OUT/short-$R-$round.txt 2>&1
        wait
        tail -1 $OUT/long-$R-$round.txt
        tail -1 $OUT/short-$R-$round.txt
    done
    docker logs lm > $OUT/gpu-scatter-$R.log 2>&1
    echo "steps with uncomputed cells in their range: $(grep -c 'RS range' $OUT/gpu-scatter-$R.log)"
    docker rm -f lm >/dev/null 2>&1
done

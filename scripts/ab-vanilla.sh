#!/usr/bin/env bash
# Vanilla llama.cpp against this fork, each at its best configuration for the card, on the same workloads:
#   short   - short prompts at rising concurrency: generation throughput and latency
#   mixed   - long documents next to short prompts: long and short requests sharing the memory
#   longest - one document longer than a vanilla slot holds (its context is split evenly across the slots)
#   ocr     - the citadel OCR battery: images, wall time and output quality
#
# The servers alternate V F V F with a fresh server per run, because a single run of each says little: the same code
# has been seen to vary by ~10% between runs on this card. Every run sends the identical workload.
# Every response is saved; the comparison at the end reports how similar the two servers' outputs are, which must stay
# at the level of batching nondeterminism.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/ab_vanilla}
MODELS=$CITADEL/data/gguf_models
IMG_FORK=nvidia/cuda:13.3.0-devel-ubuntu24.04
IMG_VANILLA=ghcr.io/ggml-org/llama.cpp:server-cuda
ROUNDS=${ROUNDS:-2}

COMMON="--model /models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf} --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k ${KV:-q8_0} --cache-type-v ${KV:-q8_0} --flash-attn on --alias lm"

mkdir -p $OUT

serve() { # $1 = vanilla|fork
    docker rm -f lm >/dev/null 2>&1
    if [ $1 = vanilla ]; then
        # the production configuration: 131072 tokens split into 4 slots of 32768
        docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $MODELS:/models:ro $IMG_VANILLA \
            $COMMON --ctx-size 131072 --parallel 4 >/dev/null
    else
        # one pool of 65536 tokens, as many requests at once as fit (at most 32)
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

bench() { # $1 = vanilla|fork, $2 = round
    local S=$1 r=$2

    serve $S
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/memory used after startup: /"

    echo "--- $S round $r: short prompts"
    python3 $SRC/scripts/bench-server-concurrency.py --levels 1 4 8 16 24 --max-tokens 256 --save $OUT/short-$S-$r.json

    echo "--- $S round $r: long documents (about 20k tokens) next to short prompts"
    python3 $SRC/scripts/bench-server-concurrency.py --levels 8 16 --long 2 --long-tokens 20000 --max-tokens 256 --save $OUT/mixed-$S-$r.json

    echo "--- $S round $r: a document of about 40k tokens, longer than a vanilla slot"
    python3 $SRC/scripts/bench-server-concurrency.py --levels 4 --long 1 --long-tokens 40000 --max-tokens 256 --save $OUT/longest-$S-$r.json

    echo "--- $S round $r: OCR battery"
    rm -rf $OUT/ocr-$S-$r; mkdir -p $OUT/ocr-$S-$r
    local t0=$(date +%s.%N)
    ( cd $CITADEL && uv run python data/model_baselines/run_ocr_full.py $OUT/ocr-$S-$r/ocr $OUT/ocr-$S-$r/ocr_images \
        > $OUT/ocr-$S-$r/run.log 2>&1 )
    echo "OCR battery wall: $(echo "$(date +%s.%N) - $t0" | bc) s, $(ls $OUT/ocr-$S-$r/ocr 2>/dev/null | wc -l) documents"

    curl -s localhost:8100/metrics | grep -v '^#' > $OUT/metrics-$S-$r.txt
    docker logs lm > $OUT/server-$S-$r.log 2>&1
    docker rm -f lm >/dev/null 2>&1
}

for r in $(seq 1 $ROUNDS); do
    bench vanilla $r
    bench fork    $r
done

echo "=== output similarity, fork against vanilla (1.000 = identical)"
python3 - "$OUT" "$ROUNDS" <<'PY'
import difflib, json, os, sys
out, rounds = sys.argv[1], int(sys.argv[2])

def sim(a, b):
    if a is None or b is None:
        return None
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def report(name, pairs):
    ok = [p for p in pairs if p is not None]
    n_missing = len(pairs) - len(ok)
    print(f"{name:26}: identical {sum(p == 1.0 for p in ok)}/{len(pairs)}, lowest {min(ok, default=float('nan')):.3f}"
          + (f", {n_missing} missing (failed on one server)" if n_missing else ""))

for r in range(1, rounds + 1):
    for name in ("short", "mixed", "longest"):
        v = json.load(open(f"{out}/{name}-vanilla-{r}.json"))
        f = json.load(open(f"{out}/{name}-fork-{r}.json"))
        for level in v:
            report(f"{name} round {r} level {level}", [sim(a, b) for a, b in zip(v[level], f[level])])

    dv, df = f"{out}/ocr-vanilla-{r}/ocr", f"{out}/ocr-fork-{r}/ocr"
    report(f"ocr round {r}", [sim(open(f"{dv}/{n}").read(), open(f"{df}/{n}").read())
                              for n in sorted(os.listdir(dv)) if os.path.exists(f"{df}/{n}")])
PY

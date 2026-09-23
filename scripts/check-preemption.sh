#!/usr/bin/env bash
# What the server does when the requests it serves together need more KV cells than --ctx-size holds: the newest slots
# are preempted (their cells freed) and recomputed from their tokens once there is room.
# The OCR load of every non-spreadsheet file in ~/Downloads/ocr_input (citadel's run_ocr_pool.py: PDF pages, images,
# embedded pictures) runs twice on the production configuration (65536 tokens, as many requests at once as the budget
# allows), each on a fresh server:
#   ref     - 5 requests at once: even 5 of the largest pages with the longest output fit, so nothing is preempted
#   load    - 48 requests at once: more than the cache holds, so the server must preempt and resume
# The outputs of the two must match up to batching nondeterminism, every request must complete, every preemption must
# resume, and no memory may grow.
#   toolong - one document of ~20k tokens with --ctx-size 16384: must be refused with an error, not crash or hang
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/preemption}
MODELS=$CITADEL/data/gguf_models
IMG=nvidia/cuda:13.3.0-devel-ubuntu24.04
MAX_PAGES=${MAX_PAGES:-10}   # pages per PDF
BUDGET=${BUDGET:-}           # e.g. 21G: on a GPU that also drives a desktop, the device memory the server may use
RUNS=${RUNS:-"ref load toolong"}   # e.g. "load" to repeat only the load run against an earlier ref

COMMON="--model /models/Qwen3.8-27B-UD-IQ3_XXS.gguf --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k f16 --cache-type-v f16 --flash-attn on --alias lm
        --rs-rollback replay --cache-ram 0 -lv 4"

mkdir -p $OUT

serve() { # $1 = label, $2 = ctx size, $3 = --parallel
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server $COMMON --ctx-size $2 --parallel $3 \
        ${BUDGET:+--vram-budget $BUDGET} >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$1.log 2>&1
        echo "$1 did not start, see $OUT/server-$1.log"
        return 1
    fi
    docker logs lm 2>&1 | grep -h -E "common_budget_params: (limited|at the requested)" | sed -E 's/.*common_budget_params: /  /' | cut -c1-160
}

finish() { # $1 = label
    curl -s localhost:8100/metrics | grep -v '^#' > $OUT/metrics-$1.txt
    docker logs lm > $OUT/server-$1.log 2>&1
    docker rm -f lm >/dev/null 2>&1
    echo "  $(grep -h -E 'preemptions_total|resumes_total' $OUT/metrics-$1.txt | tr '\n' ' ')"
    echo "  server log: $(grep -c 'preempted to free' $OUT/server-$1.log) preemptions, $(grep -c ' E ' $OUT/server-$1.log) error lines, $(grep -c 'grows past its reserve' $OUT/server-$1.log) pool growths past the reserve, $(grep -c 'out of memory' $OUT/server-$1.log) out of memory"
}

pool() { # $1 = label, $2 = requests at once
    echo "--- $1: OCR pool, $2 requests at once"
    serve $1 65536 64 || return
    rm -rf $OUT/pool-$1
    local t0=$(date +%s.%N)
    ( cd $CITADEL && uv run python data/model_baselines/run_ocr_pool.py $OUT/pool-$1 $2 $MAX_PAGES > $OUT/pool-$1.log 2>&1 )
    echo "  $(echo "$(date +%s.%N) - $t0" | bc | cut -d. -f1) s: $(tail -1 $OUT/pool-$1.log)"
    finish $1
}

case " $RUNS " in *" ref "*)  pool ref  5 ;; esac
case " $RUNS " in *" load "*) pool load 48 ;; esac

case " $RUNS " in *" toolong "*)
echo "--- toolong: one document of ~20k tokens, --ctx-size 16384"
if serve toolong 16384 4; then
    timeout 300 python3 $SRC/scripts/bench-server-concurrency.py --levels 1 --long 1 --long-tokens 20000 --max-tokens 64 --seed 7 \
        > $OUT/text-toolong.txt 2>&1
    echo "  bench exit $?: $(tail -2 $OUT/text-toolong.txt | tr '\n' ' ' | cut -c1-200)"
    echo "  server still healthy: $(curl -sf localhost:8100/health >/dev/null && echo yes || echo no)"
    grep -h -i -E "exceed|too long|context size|n_ctx" <(docker logs lm 2>&1) | tail -2 | sed 's/^/  server: /' | cut -c1-200
    finish toolong
fi
;; esac

echo "=== load against ref, per request (1.000 = identical)"
python3 - "$OUT" <<'PY'
import difflib, json, os, sys
out = sys.argv[1]
ref, load = f"{out}/pool-ref", f"{out}/pool-load"
rr = {r["label"]: r for r in json.load(open(f"{ref}/results.json"))} if os.path.exists(f"{ref}/results.json") else {}
rl = {r["label"]: r for r in json.load(open(f"{load}/results.json"))} if os.path.exists(f"{load}/results.json") else {}
for name, res in (("ref", rr), ("load", rl)):
    failed = [l for l, r in res.items() if "error" in r]
    cut = [l for l, r in res.items() if r.get("finish_reason") == "length"]
    print(f"{name:4}: {len(res)} requests, {len(failed)} failed{': ' + ', '.join(failed[:5]) if failed else ''}, {len(cut)} stopped at max tokens")
sims = []
for label in sorted(rr):
    a, b = f"{ref}/{label}.md", f"{load}/{label}.md"
    if os.path.exists(a) and os.path.exists(b):
        x, y = open(a).read(), open(b).read()
        sims.append((1.0 if x == y else difflib.SequenceMatcher(None, x, y, autojunk=False).ratio(), label))
if sims:
    print(f"identical {sum(s == 1.0 for s, _ in sims)}/{len(sims)}, above 0.95 {sum(s >= 0.95 for s, _ in sims)}/{len(sims)}")
    print("lowest: " + ", ".join(f"{label} {s:.3f}" for s, label in sorted(sims)[:6]))
PY

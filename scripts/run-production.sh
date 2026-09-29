#!/usr/bin/env bash
# The production configuration on one server, and the workloads citadel sends it:
#   battery - the 16 page OCR battery, scored against OCR_GROUND_TRUTH.md
#   pool    - every non-spreadsheet file of ~/Downloads/ocr_input (run_ocr_pool.py), 16 requests at once as citadel sends them
#   text    - short prompts at rising concurrency, and long documents next to short prompts
# and what the server did meanwhile: requests at once, generation stalls, preemptions, memory.
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/production}
MODELS=$CITADEL/data/gguf_models
IMG=nvidia/cuda:13.3.0-devel-ubuntu24.04
MODEL=${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf}
KV=${KV:-q8_0}
CTX=${CTX:-131072}
BUDGET=${BUDGET:-22G}
MAX_PAGES=${MAX_PAGES:-10}   # pages per PDF in the pool
RUNS=${RUNS:-"battery pool text"}
PARALLEL=${PARALLEL:-64}     # the most sequences at once, the budget may allow fewer
CACHE_RAM=${CACHE_RAM:-0}    # MiB of host memory for the prompt cache, 0 disables it: citadel requests share no prompts

mkdir -p $OUT

docker rm -f lm >/dev/null 2>&1
docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $SRC:/src -v $MODELS:/models:ro \
    -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server \
    --model /models/$MODEL --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024 \
    --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100 \
    --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k $KV --cache-type-v $KV --flash-attn on --alias lm \
    --ctx-size $CTX --parallel $PARALLEL --vram-budget $BUDGET --rs-rollback replay --cache-ram $CACHE_RAM -lv 4 >/dev/null
until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
    docker logs lm > $OUT/server.log 2>&1
    echo "the server did not start, see $OUT/server.log"
    exit 1
fi
echo "=== $MODEL, $KV cache, $CTX tokens, budget $BUDGET"
docker logs lm 2>&1 | grep -h -E "common_budget_params: (limited|at the requested)|in place enabled|device_memory_log" | sed -E 's/^[0-9.]+ I //' | cut -c1-200

# the workloads run in the order listed, a workload listed twice runs twice (e.g. "text pool text" shows whether the
# server is as fast after the OCR pool as before it); the files of the k-th workload end in -k
k=0
for run in $RUNS; do
    k=$((k + 1))
    case $run in
    battery)
        echo "--- [$k] OCR battery"
        rm -rf $OUT/battery-$k; mkdir -p $OUT/battery-$k
        t0=$(date +%s.%N)
        ( cd $CITADEL && uv run python data/model_baselines/run_ocr_full.py $OUT/battery-$k/ocr $OUT/battery-$k/ocr_images > $OUT/battery-$k/run.log 2>&1 )
        echo "  $(echo "$(date +%s.%N) - $t0" | bc | cut -d. -f1) s, $(ls $OUT/battery-$k/ocr 2>/dev/null | wc -l) of 16 documents"
        python3 $SRC/scripts/ocr-score.py -v $OUT/battery-$k/ocr | sed 's/^/  /'
        ;;
    pool)
        echo "--- [$k] OCR pool, 16 requests at once"
        rm -rf $OUT/pool-$k
        ( cd $CITADEL && uv run python data/model_baselines/run_ocr_pool.py $OUT/pool-$k 16 $MAX_PAGES > $OUT/pool-$k.log 2>&1 )
        echo "  $(tail -1 $OUT/pool-$k.log)"
        ;;
    text)
        echo "--- [$k] short prompts at rising concurrency"
        python3 $SRC/scripts/bench-server-concurrency.py --levels 1 4 8 16 --max-tokens 256 --save $OUT/text-short-$k.json > $OUT/text-short-$k.txt 2>&1
        tail -6 $OUT/text-short-$k.txt | sed 's/^/  /'
        echo "--- [$k] 2 long documents next to short prompts"
        python3 $SRC/scripts/bench-server-concurrency.py --levels 8 --long 2 --long-tokens 40000 --max-tokens 256 --save $OUT/text-long-$k.json > $OUT/text-long-$k.txt 2>&1
        tail -4 $OUT/text-long-$k.txt | sed 's/^/  /'
        ;;
    *)
        echo "unknown workload $run"
        ;;
    esac
done

curl -s localhost:8100/metrics | grep -v '^#' > $OUT/metrics.txt
docker logs lm > $OUT/server.log 2>&1
docker rm -f lm >/dev/null 2>&1

echo "=== the server meanwhile"
echo "  $(grep -h -E 'preemptions_total|resumes_total' $OUT/metrics.txt | tr '\n' ' ')"
echo "  low point of free device memory: $(grep device_memory_free_min $OUT/metrics.txt | awk '{printf "%.0f MiB", $2/1048576}')"
echo "  $(grep -c 'grows past its reserve' $OUT/server.log) pool growths past the reserve, $(grep -c 'out of memory' $OUT/server.log) out of memory, $(grep -c ' E ' $OUT/server.log) error lines"
grep -h "slow iteration" $OUT/server.log | grep -v "generating = 0" \
    | sed -E 's/.*slow iteration: ([0-9]+) ms.*media ([0-9]+) ms.*/\1 \2/' \
    | awk '{ n++; t += $1; m += $2; if ($1 > mx) mx = $1 } END { printf "  generation stalls: %d slow iterations, %.1f s in total (%.1f s of it media), longest %d ms\n", n, t/1000, m/1000, mx }'

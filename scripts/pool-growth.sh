#!/usr/bin/env bash
# Device memory of the server against the startup projection, on the OCR battery.
# The projection now includes the memory the backends take outside their buffers (op scratch and cuBLAS handles), sized
# from the worst-case graphs and taken at startup: the server should not grow during the battery, and a pool that grows
# past its reserve logs "grows past its reserve" with the op that made it grow.
#   vanilla-4 - vanilla at 4 requests at once (its own growth, for comparison)
#   fork-4    - this fork at 4 requests at once
#   fork-max  - this fork at up to 32 requests at once
#   fork-fill - this fork at up to 64 requests at once, so that the budget alone decides how many fit: the free memory
#               at startup is all used, and the battery must still complete
set -u

SRC=${SRC:-/home/masterkenway/Projects/llama.cpp}
VAN=${VAN:-/home/masterkenway/Projects/llama.cpp-vanilla}
CITADEL=${CITADEL:-/home/masterkenway/Projects/citadel}
OUT=${OUT:-$CITADEL/data/model_baselines/pool_growth}
MODELS=$CITADEL/data/gguf_models
IMG=nvidia/cuda:13.3.0-devel-ubuntu24.04

COMMON="--model /models/${MODEL:-Qwen3.8-27B-AP-IQ4_XS.gguf} --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024
        --chat-template-file /models/chat_template.jinja -ngl 999 --host 0.0.0.0 --port 8100
        --spec-type draft-mtp --spec-draft-n-max 2 --metrics --cache-type-k f16 --cache-type-v f16 --flash-attn on --alias lm
        --ctx-size 65536"

mkdir -p $OUT

build() { # $1 = source tree
    docker run --rm -i --gpus all -v $1:/src -w /src $IMG bash -c "apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null && git config --global --add safe.directory '*' && [ -f build/CMakeCache.txt ] || cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build --config Release -j --target llama-server" 2>&1 | tail -1
}

# once a second: time, device memory used in total, device memory of the server process alone (MiB)
# the GPU also drives the display, so the total includes the desktop; only the process column is the server
sample() { # $1 = output file
    while true; do
        echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits | grep llama-server | awk -F', ' '{ s += $2 } END { print s + 0 }')"
        sleep 1
    done >> $1
}

run() { # $1 = label, $2 = source tree, $3... = extra server args
    local label=$1 tree=$2
    shift 2
    rm -f $OUT/mem-$label.txt
    sample $OUT/mem-$label.txt &
    local sampler=$!
    docker rm -f lm >/dev/null 2>&1
    docker run -d --name lm --gpus all --ulimit core=0 -p 8100:8100 -v $tree:/src -v $MODELS:/models:ro \
        -e LD_LIBRARY_PATH=/src/build/bin $IMG /src/build/bin/llama-server $COMMON "$@" >/dev/null
    until curl -sf localhost:8100/health >/dev/null || [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; do sleep 1; done
    if [ "$(docker inspect -f '{{.State.Running}}' lm)" != "true" ]; then
        docker logs lm > $OUT/server-$label.log 2>&1
        echo "$label did not start, see $OUT/server-$label.log"
        kill $sampler
        return
    fi
    sleep 2
    echo "# ready" >> $OUT/mem-$label.txt
    docker logs lm > $OUT/startup-$label.log 2>&1

    rm -rf $OUT/ocr-$label; mkdir -p $OUT/ocr-$label
    local t0=$(date +%s.%N)
    ( cd $CITADEL && uv run python data/model_baselines/run_ocr_full.py $OUT/ocr-$label/ocr $OUT/ocr-$label/ocr_images \
        > $OUT/ocr-$label/run.log 2>&1 )
    echo "$label: OCR battery $(echo "$(date +%s.%N) - $t0" | bc) s, $(ls $OUT/ocr-$label/ocr 2>/dev/null | wc -l) documents"
    echo "# done" >> $OUT/mem-$label.txt
    kill $sampler

    curl -s localhost:8100/metrics | grep -v '^#' > $OUT/metrics-$label.txt
    docker logs lm > $OUT/server-$label.log 2>&1
    docker rm -f lm >/dev/null 2>&1
}

echo "building the fork";   build $SRC
echo "building vanilla";    build $VAN

# -lv 4 on every server: below it the ggml INFO lines, the scratch sizes among them, are not printed
[ -n "${ONLY:-}" ] || ONLY="fork-4 fork-max fork-fill"
case " $ONLY " in *" vanilla-4 "*) run vanilla-4 $VAN --parallel 4  -lv 4 ;; esac
case " $ONLY " in *" fork-4 "*)    run fork-4    $SRC --parallel 4  --rs-rollback replay --cache-ram 4096 -lv 4 ;; esac
case " $ONLY " in *" fork-max "*)  run fork-max  $SRC --parallel 32 --rs-rollback replay --cache-ram 4096 -lv 4 ;; esac
case " $ONLY " in *" fork-fill "*) run fork-fill $SRC --parallel 64 --rs-rollback replay --cache-ram 4096 -lv 4 ;; esac

# per run: what the startup projected and reserved, and whether anything grew during the battery
for label in vanilla-4 fork-4 fork-max fork-fill; do
    [ -f $OUT/server-$label.log ] || continue
    echo "=== $label"
    grep -h -E "common_budget_params: (limited|at the requested)" $OUT/server-$label.log | sed -E 's/.*common_budget_params: //' | cut -c1-200
    grep -h -E "scratch size" $OUT/server-$label.log | sed -E 's/^[0-9.]+ I //; s/ \(op scratch.*//' | sed 's/^/  /'
    grep -h "device memory: projected" $OUT/server-$label.log | sed -E 's/.*device memory: /  at startup: /'
    n=$(grep -c "grows past its reserve" $OUT/server-$label.log)
    echo "pools grown past their reserve during the battery: $n"
    grep -h "grows past its reserve" $OUT/server-$label.log | sed -E 's/^[0-9.]+ W //' | head -10 | sed 's/^/  /'
    grep -h -c "out of memory" $OUT/server-$label.log | sed 's/^/out of memory errors: /'
    echo "low point of free device memory: $(grep device_memory_free_min $OUT/metrics-$label.txt 2>/dev/null | awk '{printf "%.0f MiB", $2/1048576}')"
    # the server process alone, from nvidia-smi: when ready, at its peak during the battery; the rest of the device is the desktop
    awk '/^# ready/ { phase = 1; next } /^# done/ { phase = 2; next }
         phase == 1 && ready == "" && $3 > 0 { ready = $3 }
         phase == 1 { if ($3 > peak) peak = $3; o = $2 - $3; if (omin == "" || o < omin) omin = o; if (o > omax) omax = o }
         END { printf "server process (nvidia-smi): %d MiB when ready, %d MiB at peak, grew by %d MiB during the battery; other processes on the device: %d to %d MiB\n", ready, peak, peak - ready, omin, omax }' $OUT/mem-$label.txt
    python3 $SRC/scripts/ocr-score.py $OUT/ocr-$label/ocr
done > $OUT/summary.txt
cat $OUT/summary.txt

#!/usr/bin/env bash
# Runs the recurrent rollback tests on the tiny generated hybrid models, with --rs-rollback snapshot and replay,
# on the CPU and on the GPU. Takes a few minutes, mostly the build.
set -u

SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/home/masterkenway/Projects/citadel/data/model_baselines/replay_ab/unit}
IMG=${IMG:-nvidia/cuda:13.3.0-devel-ubuntu24.04}

mkdir -p $OUT

docker run --rm -i --gpus all --ulimit core=0 -v $SRC:/src -v $OUT:/out -w /src $IMG bash -c '
    apt-get update >/dev/null && apt-get install -y cmake build-essential git >/dev/null
    git config --global --add safe.directory /src
    cmake --build build --config Release -j --target llama-server test-llama-archs test-recurrent-state-rollback test-gdn-replay 2>&1 | tail -2

    export LD_LIBRARY_PATH=/src/build/bin
    rb() { [ "$1" = 1 ] && echo replay || echo snapshot; }
    M=/src/build/tests/test-models
    mkdir -p $M
    ./build/bin/test-llama-archs -o $M > /out/generate.log 2>&1 || { echo "model generation failed, see generate.log"; exit 1; }
    ls $M | grep -E "qwen35|nemotron" | tr "\n" " "; echo

    ./build/bin/test-gdn-replay | tail -1

    for dev in cpu gpu; do
        for R in 0 1; do
            if [ $dev = cpu ]; then args="-dev none -ngl 0"; else args="-ngl 99"; fi
            for model in qwen35-dense nemotron_h-dense; do
                log=/out/rollback-$model-$dev-$R.log
                ./build/bin/test-recurrent-state-rollback -m $M/$model.gguf $args --rs-rollback $(rb $R) > $log 2>&1
                res=$?
                printf "%-18s %s %-8s: exit %d | %s\n" $model $dev $(rb $R) $res \
                    "$(grep -E "test_shared_pending|replay matched|restored successfully|mismatch|failed" $log | sed -E "s/^.*: //" | tr "\n" ";" | cut -c1-200)"
            done
        done
    done
'

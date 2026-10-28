#!/bin/bash
# Profiles an inmoloc server image with Nsight Systems under the mixed text workload at 128 concurrent requests:
# 20 s of the steady state (from 60 s after launch), then nsys stops the server and the client run is cut short.
#
#   scripts/profile/profile-server.sh <tag> [name] [extra server args...]
#
# e.g. scripts/profile/profile-server.sh port15 port15-replay --rs-rollback replay
# writes <name>.nsys-rep, <name>.sqlite and kernel/API summaries to ~/Projects/bench-results/profile
set -eo pipefail
here=$(dirname "$(readlink -f "$0")")
cd "$here/../.."
out=~/Projects/bench-results/profile
mkdir -p "$out"

tag=${1:?usage: profile-server.sh <tag> [name] [extra server args...]}
name=${2:-$tag}
shift; [ $# -gt 0 ] && shift
image=inmoloc:$tag
port=8131

docker build --build-arg BASE="$image" -t "$image-nsys" "$here"
docker rm -f "prof-$name" >/dev/null 2>&1 || true
docker run -d --name "prof-$name" --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "$port:$port" -v ~/models:/models:ro -v "$out:/out" --entrypoint nsys "$image-nsys" \
  profile -t cuda,nvtx,osrt -s none --cpuctxsw none --cuda-graph-trace node --delay 60 --duration 20 \
    -o "/out/$name" -f true /app/local-inference-server \
    --model /models/Qwen3.8-27B-AP-Q4_K_XL.gguf --host 0.0.0.0 --port "$port" -ngl 999 --metrics \
    -fa on --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --cache-ram 0 \
    --vram-budget 22826M --ctx-typical 2048 "$@"
until curl -sf "localhost:$port/health" >/dev/null; do sleep 2; done
python3 scripts/bench-server-concurrency.py --url "http://localhost:$port" --levels 128 --seed 0 \
  --prompt-tokens 64 --prompt-tokens-max 2048 --max-tokens 32 --max-tokens-max 512 || true
docker wait "prof-$name" >/dev/null
docker logs "prof-$name" > "$out/$name-server.log" 2>&1
docker rm "prof-$name" >/dev/null
docker run --rm -v "$out:/out" --entrypoint nsys "$image-nsys" stats -f csv -o "/out/$name" \
  -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "/out/$name.nsys-rep" >/dev/null
ls -la "$out"/"$name"*

#!/bin/bash
# Profiles an inmoloc server image with Nsight Systems under the mixed text workload at 128 concurrent requests:
# 20 s of the steady state (from 60 s after launch), then nsys stops the server and the client run is cut short.
#
#   scripts/profile/profile-server.sh <tag> [name] [extra server args...]
#
# e.g. scripts/profile/profile-server.sh port15 port15-replay --rs-rollback replay
# writes <name>.nsys-rep, <name>.sqlite and kernel/API summaries to ~/Projects/bench-results/profile
#
# PROFILE_CPU=1 also samples the server's CPU call stacks (where the CPU spends the gaps in which the GPU waits); the
# host blocks unprivileged perf (kernel.perf_event_paranoid = 3), so the container then runs with SYS_ADMIN
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

docker_cpu=()
nsys_cpu=(-s none --cpuctxsw none)
if [ "${PROFILE_CPU:-0}" = 1 ]; then
  docker_cpu=(--cap-add SYS_ADMIN --security-opt seccomp=unconfined)
  nsys_cpu=(-s process-tree --cpuctxsw process-tree)
fi

# Nsight Systems lives in its own image and in the nsight-systems volume, mounted into the server image: installed once,
# not again for every server image
tools=inmoloc-nsys-tools
nsys_dir=/opt/nvidia/nsight-systems
docker build -q -t "$tools" "$here" >/dev/null
if ! docker run --rm -v nsight-systems:/n --entrypoint test "$tools" -x /n/current/bin/nsys; then
  docker run --rm -v nsight-systems:/n --entrypoint cp "$tools" -a "$nsys_dir/." /n/
fi

docker rm -f "prof-$name" >/dev/null 2>&1 || true
docker run -d --name "prof-$name" --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 "${docker_cpu[@]}" \
  -p "$port:$port" -v ~/models:/models:ro -v "$out:/out" -v nsight-systems:"$nsys_dir":ro \
  --entrypoint "$nsys_dir/current/bin/nsys" "$image" \
  profile -t cuda,nvtx,osrt "${nsys_cpu[@]}" --cuda-graph-trace node --delay 60 --duration 20 \
    -o "/out/$name" -f true /app/local-inference-server \
    --model /models/Qwen3.8-27B-AP-Q4_K_XL.gguf --host 0.0.0.0 --port "$port" -ngl 999 --metrics \
    -fa on --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --cache-ram 0 \
    --vram-budget 22826M --ctx-typical 2048 "$@"
until curl -sf "localhost:$port/health" >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "prof-$name" 2>/dev/null)" != true ]; then
    echo "prof-$name exited before the server was ready:" >&2
    docker logs --tail 20 "prof-$name" >&2
    docker rm "prof-$name" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 2
done
python3 scripts/bench-server-concurrency.py --url "http://localhost:$port" --levels 128 --seed 0 \
  --prompt-tokens 64 --prompt-tokens-max 2048 --max-tokens 32 --max-tokens-max 512 || true
docker wait "prof-$name" >/dev/null
docker logs "prof-$name" > "$out/$name-server.log" 2>&1
docker rm "prof-$name" >/dev/null
docker run --rm -v "$out:/out" --entrypoint "$nsys_dir/current/bin/nsys" "$tools" stats -f csv -o "/out/$name" \
  -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "/out/$name.nsys-rep" >/dev/null
ls -la "$out"/"$name"*

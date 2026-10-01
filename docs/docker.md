# Docker

## Images

The images are published to the GitHub Container Registry on every release:

| Image | Contents |
| --- | --- |
| `ghcr.io/proafxin/inmoloc:latest`, `:<version>` | `local-inference-server` built with CUDA 12.8, for NVIDIA GPUs from Turing (RTX 20xx, T4) to Blackwell (RTX 50xx) |
| `ghcr.io/proafxin/inmoloc:cpu`, `:<version>-cpu` | `local-inference-server` for the CPU only |

Both images start the server, listen on port 8080 on all interfaces and include the web UI. The CUDA image needs an NVIDIA driver of version 570 or newer on the host, and the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

## Running the server

Serve a model from a host directory with a 128k-token context shared by all requests, a q8_0 KV cache and a 22 GiB memory budget:

```sh
docker run --gpus all -p 8080:8080 -v /path/to/models:/models ghcr.io/proafxin/inmoloc:latest \
    -m /models/model.gguf -ngl 999 --flash-attn on \
    --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --vram-budget 22G --parallel 64
```

The arguments after the image name are passed to `local-inference-server`; see [the server documentation](../tools/server/README.md). Every argument can also be given as an environment variable, for example `-e LOCAL_INFERENCE_ARG_CTX_SIZE=131072`.

The memory budget is lowered to the device memory that is free when the server starts, so other programs on the GPU reduce the number of requests it runs at once. The log states the number and what fills the budget.

With Docker Compose:

```yml
services:
  local-inference-server:
    image: ghcr.io/proafxin/inmoloc:latest
    ports:
      - 8080:8080
    volumes:
      - /path/to/models:/models:ro
    command: >
      -m /models/model.gguf -ngl 999 --flash-attn on
      --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --vram-budget 22G --parallel 64
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Building the images

```sh
docker build -t inmoloc:cuda -f .devops/cuda.Dockerfile .
docker build -t inmoloc:cpu  -f .devops/cpu.Dockerfile .
```

The default target is the server; `--target light` builds an image with `local-inference-cli` instead. Build arguments of the CUDA image:

- `CUDA_VERSION`, default `12.8.1`: must be supported by the host driver.
- `CUDA_DOCKER_ARCH`, default `75-real;80-real;86-real;89-real;90-real;120a-real`: the GPU architectures to compile for. Building for your GPU alone is much faster, for example `--build-arg CUDA_DOCKER_ARCH=120a-real` for an RTX 50xx or `86-real` for an RTX 30xx.

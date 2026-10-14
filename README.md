# inmoloc

<div align="center">

<b>Concurrent LLM serving on your own GPU, with the startup time and memory footprint of a local runtime.</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

inmoloc (Inferencia de Modelos Locales) serves many requests at once from one GPU. It starts in seconds, fits the number of concurrent requests to the memory you give it, and keeps each request's cost tied to its own context. It runs GGUF models, including vision models, through an OpenAI-compatible HTTP API.

inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends. It is maintained as its own project; see [differences from llama.cpp](docs/differences-from-llama.cpp.md) and [NOTICE](NOTICE).

## Features

- **Shared KV cache**: all requests share one cache of `--ctx-size` tokens, and each request takes cells only for the tokens it holds, up to the model's training context.
- **Memory budget**: at startup the server measures the weights, KV cache, per-request state, compute buffers, draft model, vision encoder and backend scratch memory, and runs the most requests at once that fit `--vram-budget`. Memory does not grow while serving.
- **Preemption**: when the cache is full, the newest requests are paused and recomputed later, and the others continue.
- **Per-request attention** (CUDA): each request reads only its own KV cells, so its cost grows with its own context, not with other requests. Quantized q8_0 and q4_0 caches are read directly.
- **Prefix sharing**: a new request starts from the cached cells of another request with the same prefix, without copying them.
- **Cost-based prompt scheduling**: long prompts are processed in chunks sized by their measured cost, so requests that are generating are not held up.
- **Asynchronous vision encoder**: images are encoded on their own thread and lower-priority GPU stream while other requests generate.
- **Replay rollback for speculative decoding**: on hybrid gated delta-net models such as Qwen3.5, memory per request stays nearly the same at any draft depth.
- **Low CPU load while waiting for the GPU** (CUDA): with `--gpu-wait block` the server thread sleeps instead of spinning a CPU core.
- **Observability**: the startup log shows the memory projection and what fills it; Prometheus metrics cover KV cells, preemptions, prefix sharing, attention cost and free device memory; `--slow-loop-ms` logs slow server iterations.

## Quick start

Run the published image (NVIDIA driver 570 or newer, Turing to Blackwell GPUs; see [Docker](docs/docker.md)):

```sh
docker run --gpus all -p 8080:8080 -v /path/to/models:/models ghcr.io/proafxin/inmoloc:latest \
    -m /models/model.gguf -ngl 999 --flash-attn on --ctx-size 131072 --vram-budget 22G --parallel 64
```

Or build with CUDA (see [the build guide](docs/build.md) for other backends):

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target local-inference-server
```

Serve a model with a 128k-token context shared by all requests, a q8_0 KV cache and a 22 GiB memory budget:

```sh
build/bin/local-inference-server -m model.gguf --mmproj mmproj.gguf -ngl 999 --flash-attn on \
    --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 \
    --vram-budget 22G --parallel 64 --gpu-wait block --metrics --port 8100
```

The log states how many requests run at once and what the budget holds, for example:

```text
limited by the budget: 14 sequences, 131072 tokens of context: model 13061 + context 6763 + compute 297 + draft 725 + other 1662 = 22508 MiB of 22528 MiB
```

## Server options

The options below are specific to inmoloc; all other options are as in llama.cpp, see [the server documentation](tools/server/README.md).

| Option | Default | Description |
| --- | --- | --- |
| `--vram-budget SIZE` | free device memory | Device memory the server may use, summed over the devices the model runs on (e.g. `22G`). |
| `--parallel N` | | Upper limit on requests at once; the budget may lower it. |
| `--prompt-cap N` | `-1` | Prompt tokens per iteration while requests generate, weighted by cost. `-1`: fill one ubatch, `0`: no cap. |
| `--prefix-share`, `--no-prefix-share` | enabled | Let a new request start from another request's cached prefix. |
| `--prefix-share-min N` | `256` | Least number of tokens a shared prefix must save. |
| `--rs-rollback {snapshot,replay}` | `snapshot` | How recurrent state rolls back rejected speculative tokens. `replay` applies to gated delta-net models. |
| `--gpu-wait {spin,yield,block}` | driver | How a thread waits for the GPU (CUDA). `block` sleeps and frees the CPU core. |
| `--slow-loop-ms N` | `1000` | Log a time breakdown of server iterations slower than N ms. `0`: off. |

When `--ctx-size` or `--gpu-layers` is not given, `--fit` chooses them first with its own margin (`--fit-target`).

## Measured

On one RTX 5090 laptop GPU (24 GiB) with Qwen3.8-27B IQ4_XS, MTP speculative decoding, the vision encoder, a q8_0 cache of 131072 tokens and `--vram-budget 22G`:

- **14 requests at once**, and serving in about 5 seconds from start.
- **No memory growth while serving**: device memory stayed within 6 MiB over 104 OCR requests, 16 at a time.
- **Long prompts next to other requests**: two 56k-token documents were processed at 680 to 730 tokens per second each, with short requests generating alongside.
- **q8_0 cache quality**: at 32k, 64k and 110k tokens of context, the q8_0 cache answered retrieval questions at 10%, 50% and 90% depth exactly as the f16 cache did, token for token (`scripts/kv-quality.py`).
- **CPU temperature with `--gpu-wait block`**: on an MSI Vector 16 HX AI, the CPU averaged below 65 degC while serving, against about 100 degC with the driver's default.

## Compared with llama.cpp and vLLM

llama.cpp at commit 56381e407 (2026-09-12); vLLM as of 2026-10-15.

| | llama.cpp | vLLM | inmoloc |
| --- | --- | --- | --- |
| Startup | Loads the model | Profiles memory, captures CUDA graphs, compiles the model | Loads the model after short memory dry runs |
| GPU memory taken | What the model and context need | A fixed fraction (`--gpu-memory-utilization`) | What the model and context need, within `--vram-budget` |
| KV cache | `--parallel` fixed parts by default, shared with `--kv-unified` | Shared pool of blocks | Shared, per token |
| Cache full | Every active request fails | Newest requests preempted | Newest requests preempted |
| Attention reads | All used cells | Own blocks | Own cells (CUDA) |
| Shared prefixes | Own slot only | Across requests | Across requests |
| Quantized KV cache | q8_0, q4_0 and others | fp8 | q8_0, q4_0 and others |
| GGUF models | Native | Experimental plugin | Native |
| Multi-GPU | Layer and row split | Tensor and pipeline parallel | Layer and row split |

## Compatibility

- **Models**: GGUF models supported by llama.cpp at the base commit, see [docs/models.md](docs/models.md).
- **API**: OpenAI-compatible chat completions, completions and embeddings, see [the server documentation](tools/server/README.md).
- **Backends**: all llama.cpp backends build and run. Per-request attention and `--gpu-wait` are CUDA only; other backends attend over the whole cache with a mask.
- **Multiple GPUs**: llama.cpp's multi-GPU modes work (see [Multi-GPU](docs/multi-gpu.md)); the budget is summed over devices, and multi-GPU serving is not tested yet.
- **Coming from llama.cpp**: executables are named `local-inference-*` (unified binary: `local-inference`), environment variables start with `LOCAL_INFERENCE_`, metrics start with `local_inference:`, and `--no-kv-unified` is not supported. The C API is source compatible; bindings that copy the `llama_context_params` layout need updating.

## Documentation

- [Server](tools/server/README.md)
- [Build](docs/build.md)
- [Docker](docs/docker.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Models](docs/models.md)
- [Differences from llama.cpp](docs/differences-from-llama.cpp.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

MIT, see [LICENSE](LICENSE). inmoloc is based on llama.cpp by the ggml authors; see [NOTICE](NOTICE). Third-party components keep their own licenses, see [licenses/](licenses/) and [vendor/](vendor/).

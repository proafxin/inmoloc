# inmoloc

<div align="center">

<b>Serve as many requests as your VRAM budget fits elastically</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

inmoloc (Inferencia de Modelos Locales) serves many requests at once on GPUs. It starts in seconds, fits the number of concurrent requests to the memory you give it, and keeps each request's cost tied to its own context. It runs GGUF models, including vision models, through an OpenAI-compatible HTTP API.

inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends. It is maintained as its own project; see [differences from llama.cpp](docs/differences-from-llama.cpp.md) and [NOTICE](NOTICE).

## Why inmoloc exists

llama.cpp and vLLM make you choose fixed sizes up front, and a wrong choice either wastes memory or fails to load:

- **llama.cpp needs `--parallel` and `--ctx-size` chosen by hand.** It pre-allocates the whole KV cache up front (`-c × --parallel`) on top of all layers, so on a shared GPU it either OOMs at load or you must cut the context window and concurrency until it fits — with no way to use what other services leave free. Finding a working `-c`/`--parallel` pair is trial and error: too large OOMs at load, too small wastes memory, and MTP makes the guess even harder. It retries constantly instead of queuing, which burns CPU. It stops at 256 sequences, splits its cache into fixed parts by default, and when the cache fills every active request fails. Even `--kv-unified` is a hard failure: requests beyond the unified cache fail instead of queuing. Every request pays for the cells of all the others through attention. MTP adds to the concurrency cost: every extra slot enlarges the shared context window, so it takes more VRAM.
- **vLLM reserves a fixed fraction of GPU memory** (`--gpu-memory-utilization`) before it serves, uses an arrival-order queue (small requests wait behind large ones), preempts requests that then recompute, and profiles/captures graphs/compiles the model before it serves — slow startup and extra VRAM. On Qwen3.8 27B, its [recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) lists NVFP4 at 24.6 GiB, and a 32 GB RTX 5090 fits it only with CUDA graphs off. GGUF models load only through a plugin that vLLM calls [highly experimental and under-optimized](https://docs.vllm.ai/en/latest/features/quantization/gguf/). MTP has been a problem there too: open PRs show it is still hard to use MTP without heavy VRAM cost. Measured on a 24 GiB GPU: with a 4-bit build of Qwen3.8 27B (INT4 W4A16) and MTP drafting, vLLM 0.33.0 does not start, because the draft model runs out of memory while loading; inmoloc serves the same model at Q4_K_XL with MTP drafting 2 within a 22826 MiB budget, at 387 tok/s with 64 concurrent requests and none failed.
- **inmoloc assumes a single model is fully loaded on the GPU** and does not serve from the CPU. Within that constraint, you give it a memory budget (`--vram-budget`) and an output limit (`--max-output-tokens`), and it sizes itself. It loads the model, measures what the weights and overhead actually take, then sizes the KV cache from what remains — so as many concurrent requests run as fit, nothing is set aside for requests that have not arrived, running requests never run out of memory, new ones wait, and the earliest request that fits starts. It starts in seconds, with no dry runs, replaces the expensive retry with an efficient queue, shares prefixes across requests, charges each request's attention cost only to its own cells, keeps MTP draft depth nearly free, and holds recurrent state per request flat at any draft depth with replay rollback.

| | llama.cpp | vLLM | inmoloc |
| --- | --- | --- | --- |
| Sizing | fixed `-c`/`--parallel` up front | fixed GPU-memory fraction | `--vram-budget` after measuring |
| Concurrency limit | `--parallel`, max 256 | configured limit | what fits in the budget |
| Cache full | every active request fails | newest requests preempted | running ones never run out |
| Waiting | constant retry, burns CPU | arrival-order queue | earliest request that fits starts |
| Attention cost | every request pays for all others | own blocks | each request pays for its own cells only |
| Prefix sharing | within a slot only | across requests | across requests |
| Model support | wide, all GGUF | more limited | all llama.cpp GGUF models |

inmoloc solves these simultaneously — fixed upfront sizing, CPU-burning retry, no scaling, preemption, slow startup, and MTP cost — with one memory budget and one output limit, and it loads the same models without any changes to them and gets much higher performance.

## Features

llama.cpp and vLLM are as of the versions in [the comparison](#compared-with-llamacpp-and-vllm). The models that use elastic memory are listed under [Compatibility](#compatibility).

- **Memory stays within the budget.** Everything the server needs is counted against `--vram-budget`, with no margin to tune, and memory does not grow while serving.
- **Images do not stop generation.** Other requests keep generating while images are encoded.
- **Speculative decoding does not multiply recurrent state.** On hybrid gated delta-net models such as Qwen3.5, memory per request stays nearly the same at any draft depth.
- **Waiting for the GPU does not cost a CPU core** (CUDA), with `--gpu-wait block`.
- **Models without elastic memory keep running when the cache fills.** The newest request pauses, the others continue, and it resumes where it stopped.
- **You can see where memory and time go.** The startup log shows what fills the budget, and Prometheus metrics and `--slow-loop-ms` show why requests wait and what is slow.

## Quick start

Run the published CUDA image (built with CUDA 12.8, so it needs NVIDIA driver 570 or newer, like llama.cpp's; see [Docker](docs/docker.md)):

```sh
docker run --gpus all -p 8080:8080 -v /path/to/models:/models ghcr.io/proafxin/inmoloc:latest \
    -m /models/model.gguf -ngl 999 --max-output-tokens 512 --vram-budget 22G
```

Or build it yourself, for any hardware llama.cpp supports (see [the build guide](docs/build.md) for other backends and older CUDA versions):

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target local-inference-server
```

Serve a model with a q8_0 KV cache and a 22 GiB memory budget. Elastic memory needs the most tokens a request may generate, and neither a context size nor a number of requests:

```sh
build/bin/local-inference-server -m model.gguf --mmproj mmproj.gguf -ngl 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 --max-output-tokens 512 \
    --vram-budget 22G --gpu-wait block --metrics --port 8100
```

With MTP drafting on a gated delta-net model, use replay rollback. The startup log states what the elastic memory takes and which fixed costs the budget holds:

```sh
build/bin/local-inference-server -m qwen3.8.gguf -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --rs-rollback replay --max-output-tokens 512 --vram-budget 22826M --port 8100
```

## Server options

The options below are specific to inmoloc. `--ctx-size`, `--parallel`, `--context-shift`, `--n-predict`, `--fit`, `--fit-target` and `--fit-ctx` are removed from the server, and the other options are as in llama.cpp; see [the server documentation](tools/server/README.md).

| Option | Default | Description |
| --- | --- | --- |
| `--vram-budget SIZE` | free device memory | Device memory the server may use, summed over the devices the model runs on (e.g. `22G`). |
| `--max-output-tokens N` | required with elastic memory | The most tokens a request generates. A request's `max_tokens` is capped at it, and a request without `max_tokens` uses it. |
| `--ubatch-size N` | `512` | The most tokens in one GPU step. It sets the compute memory of the model and of an MTP draft, which are fixed costs of the budget, and not how many requests run. |
| `--prefix-share`, `--no-prefix-share` | enabled | Let a new request start from another request's cached prefix. |
| `--prefix-share-min N` | `256` | Least number of tokens a shared prefix must save. |
| `--rs-rollback {snapshot,replay}` | `snapshot` | How recurrent state rolls back rejected speculative tokens. `replay` applies to gated delta-net models and takes less memory per request. |
| `--gpu-wait {spin,yield,block}` | driver | How a thread waits for the GPU (CUDA). `block` sleeps and frees the CPU core. |
| `--slow-loop-ms N` | `1000` | Log a time breakdown of server iterations slower than N ms. `0`: off. |

Unless `--gpu-layers` says otherwise, all layers go on the device, and a model that does not fit there fails at load instead of running partly on the CPU. `--flash-attn` does not apply to elastic memory. When a budget is more than the device has free, 128 MiB of the device stay free, since allocations fail in the last of its memory.

## Measured

Same GPU (RTX 5090 laptop, 24 GiB), Kvark-Sparkle-1.0-27B-Q4_K_M, MTP drafting 2, q8_0 KV cache, `--max-output-tokens 512`, 512 output tokens per request, with prompts of 128 tokens in the 23G runs and 64 tokens in the 18G runs. inmoloc sizes itself from `--vram-budget`; llama.cpp needs `-c`/`--parallel` tuned by hand and OOMs at load if guessed too high. Each server ran alone on the GPU in its own run — neither number shares VRAM with another service.

### 18G budget per server

| Requests | inmoloc wall s | inmoloc agg tok/s | llama.cpp `-c 4096 --parallel 6` wall s | llama.cpp agg tok/s | Failed (both) |
|----------|----------------|-------------------|-----------------------------------------|---------------------|---------------|
| 1 | 11.3 | 45.4 | 12.2 | 42.0 | 0 |
| 2 | 9.6 | 106.9 | 22.3 | 46.0 | 0 |
| 4 | 11.7 | 175.6 | 12.2 | 167.8 | 0 |
| 8 | 23.0 | 178.2 | 37.2 | 110.1 | 0 |
| 16 | 39.9 | 205.4 | 52.0 | 157.6 | 0 |
| 32 | 77.8 | 210.6 | 100.1 | 163.7 | 0 |
| 64 | 151.9 | 215.8 | 185.9 | 176.2 | 0 |

### 23G budget per server

| Requests | inmoloc wall s | inmoloc agg tok/s | llama.cpp `-c 8192 --parallel 8` wall s | llama.cpp agg tok/s | Failed (both) |
|----------|----------------|-------------------|-----------------------------------------|---------------------|---------------|
| 1 | 11.3 | 45.4 | 12.8 | 39.9 | 0 |
| 2 | 9.9 | 103.0 | 9.8 | 104.1 | 0 |
| 4 | 11.9 | 172.7 | 12.8 | 160.6 | 0 |
| 8 | 15.4 | 266.4 | 18.8 | 217.7 | 0 |
| 16 | 24.3 | 336.7 | 38.2 | 214.5 | 0 |
| 32 | 49.5 | 331.1 | 76.6 | 214.0 | 0 |
| 64 | 92.2 | 355.3 | 159.3 | 205.7 | 0 |
| 128 | 183.7 | 356.8 | 311.0 | 210.7 | 0 |

Consequences:

- inmoloc scales with more budget (216 → 357 agg tok/s at 64–128 concurrency); llama.cpp stays flat (~176 → ~211) because it pre-allocates `-c × --parallel` up front and hits a fixed slot ceiling. More GPU does not buy more concurrency.
- A single inmoloc request can use up to ~164k tokens of context (n_ctx_slot = 262144) within the same 23G budget. llama.cpp at its largest manual setting (`-c 8192 --parallel 8`) gives each slot only 1024 tokens — getting a larger single context window requires `--parallel 1`, which makes serving sequential and is not worth benchmarking. inmoloc handles mixed context windows elastically without any of this.
- Small-prompt benchmarks mask this ceiling; inmoloc's advantage grows with context length and concurrency because the KV cache is shared elastically instead of split into fixed slots.

### Serving

No memory growth while serving; memory stays within `--vram-budget` with no margin to tune. Startup is seconds, with no dry runs or graph compilation. A shared prefix is computed once across requests. Waiting for the GPU does not cost a CPU core (`--gpu-wait block`). Images do not stop generation — other requests keep generating while images are encoded. Speculative decoding does not multiply recurrent state on hybrid gated delta-net models such as Qwen3.5; memory per request stays nearly flat at any draft depth.

## Compatibility

- **Models**: GGUF models supported by llama.cpp at the base commit, see [docs/models.md](docs/models.md).
- **API**: OpenAI-compatible chat completions, completions and embeddings, see [the server documentation](tools/server/README.md).
- **Hardware and backends**: all of llama.cpp's backends build, but serving is for GPUs: elastic memory needs all layers on one device for now, is tested on CUDA only, and does not serve from the CPU; prompts are read faster on Turing or newer, whose tensor core kernels read them through one list. Per-request attention and `--gpu-wait` are NVIDIA only (per-request attention for prompt chunks needs Turing or newer).
- **Elastic memory by model**: attention models with one KV cache, sliding-window models, recurrent models and hybrid models use it, including hybrid models with sliding-window caches. K-only MLA models are not tested on it yet. Models with sparse attention caches (MSA, DSA, DSV4) and hybrid models with indexer caches use a KV cache and per-request state sized at startup.
- **Multiple GPUs**: llama.cpp's multi-GPU modes work (see [Multi-GPU](docs/multi-gpu.md)); the budget is summed over devices, and multi-GPU serving is not tested yet.

## Coming from llama.cpp

`llama-server` command lines need the new name and `--max-output-tokens`, and those with `--ctx-size`, `--parallel`, `--context-shift`, `--n-predict` or the `--fit` options need those removed. To let inmoloc size the server for you, replace the fixed sizes with a memory budget:

```sh
# llama.cpp
llama-server -m model.gguf -ngl 999 --flash-attn on --ctx-size 65536 --parallel 16 --kv-unified

# inmoloc
local-inference-server -m model.gguf -ngl 999 --max-output-tokens 512 --vram-budget 22G
```

| llama.cpp | inmoloc |
| --- | --- |
| `llama-server`, `llama-cli`, ... | `local-inference-server`, `local-inference-cli`, ...; unified binary `local-inference` |
| `--ctx-size N`: the KV cache size | Removed from the server: `--vram-budget` sizes the memory |
| `--parallel N`: the number of slots | Removed from the server: as many requests run at once as fit |
| `--kv-unified-per-slot N` | `--vram-budget` sizes the cache |
| `--kv-unified` | Always on; `--no-kv-unified` is refused |
| `--n-predict N`: the default output of a request, which the request can override | Removed from the server: `--max-output-tokens` is the limit, and what a request without `max_tokens` uses |
| `--gpu-layers` auto: `--fit` chooses how many layers fit | All layers go on the device, and a model that does not fit there fails at load |
| `--context-shift` | Removed from the server: a request stays within the context window |
| `--flash-attn` | No effect with elastic memory |
| `--fit`, `--fit-target`, `--fit-ctx` | Removed from the server: `--vram-budget` is the limit, with no margin |
| `LLAMA_ARG_*` environment variables | `LOCAL_INFERENCE_ARG_*` |
| `llamacpp:*` metrics | `local_inference:*` |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` | `ghcr.io/proafxin/inmoloc:latest` |

The HTTP API, GGUF models and chat templates are the same. The C API has additions, and `llama_max_parallel_sequences()` is removed since there is no fixed limit; bindings that copy the `llama_context_params` layout need updating. All option changes are listed in [differences from llama.cpp](docs/differences-from-llama.cpp.md#options).

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

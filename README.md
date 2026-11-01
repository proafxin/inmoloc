# inmoloc

<div align="center">

<b>Serve as many requests as your VRAM budget fits elastically</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

inmoloc (Inferencia de Modelos Locales) serves many requests at once on GPUs. It starts in seconds, fits the number of concurrent requests to the memory you give it, and keeps each request's cost tied to its own context. It runs GGUF models, including vision models, through an OpenAI-compatible HTTP API.

inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends. It is maintained as its own project; see [differences from llama.cpp](docs/differences-from-llama.cpp.md) and [NOTICE](NOTICE).

## Features

llama.cpp and vLLM are as of the versions in [the comparison](#compared-with-llamacpp-and-vllm). The models that use elastic memory are listed under [Compatibility](#compatibility).

- **No slot count or context size to guess.** llama.cpp needs `--parallel` and `--ctx-size` chosen by hand, and vLLM a fraction of GPU memory, and a wrong choice wastes memory or fails requests. inmoloc needs a memory budget and an output limit.
- **No fixed limit on requests at once.** llama.cpp stops at 256 sequences. With elastic memory the only limit is the memory that requests take, and nothing is set aside for requests that have not arrived.
- **Long and short requests share all the memory.** llama.cpp splits its cache into fixed parts by default, so a long request cannot use what another part leaves idle. With elastic memory, any mix of requests fits as long as its total size does.
- **A full memory does not fail or undo running requests.** When llama.cpp's cache fills, every active request fails, and vLLM preempts requests that then have to recompute. With elastic memory, running requests never run out of memory, and new ones wait.
- **Large requests do not hold up small ones.** In an arrival-order queue like vLLM's, smaller requests wait behind a large one. inmoloc starts the earliest request that fits.
- **Fast startup.** vLLM profiles memory, captures graphs and compiles the model before it serves. inmoloc starts in seconds.
- **Memory stays within the budget.** Everything the server needs is counted against `--vram-budget`, with no margin to tune, and memory does not grow while serving.
- **Requests do not slow each other down through attention** (CUDA). In llama.cpp each request pays for the cells of all the others. In inmoloc a request's cost follows its own context.
- **A shared prefix is computed once.** llama.cpp reuses a prefix only within a slot. inmoloc reuses it across requests.
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

### Generation

On one RTX 5090 laptop GPU (24 GiB) with Qwen3.8-27B, MTP drafting of 2 tokens, replay rollback, a q8_0 cache and `--vram-budget 22826M`, N requests were sent at once, each with its own text of about 64 tokens and each generating 512 tokens, one run per level. Time to first token counts from when the server starts processing a request; the wait before that is listed apart.

| Requests | Tokens/s | Tokens/s per request | Time to first token | Wait | Median between tokens | Longest gap | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 46.4 | 47.2 | 0.21 s | 0.01 s | 47 ms | 50 ms | 0 |
| 2 | 69.5 | 36.8 | 0.32 s | 0.00 s | 64 ms | 131 ms | 0 |
| 4 | 173.0 | 48.5 | 0.51 s | 0.08 s | 47 ms | 283 ms | 0 |
| 8 | 273.2 | 39.0 | 1.09 s | 0.12 s | 57 ms | 559 ms | 0 |
| 16 | 382.3 | 28.0 | 1.79 s | 0.42 s | 81 ms | 896 ms | 0 |
| 32 | 369.7 | 23.4 | 2.30 s | 8.10 s | 110 ms | 1500 ms | 0 |
| 64 | 386.6 | 19.4 | 1.46 s | 28.07 s | 116 ms | 1628 ms | 0 |

### llama.cpp, 8 slots

The same requests on the same GPU, model, cache type and MTP drafting, with the `server-cuda` image of 2026-11-01 and `-c 65536 --parallel 8 --n-predict 512`:

| Requests | Tokens/s | Tokens/s per request | Time to first token | Wait | Median between tokens | Longest gap | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 34.5 | 45.5 | 3.60 s | 0.03 s | 47 ms | 304 ms | 0 |
| 2 | 64.6 | 35.1 | 0.44 s | 0.00 s | 67 ms | 133 ms | 0 |
| 4 | 150.5 | 43.1 | 0.73 s | 0.13 s | 51 ms | 461 ms | 0 |
| 8 | 216.6 | 31.2 | 1.39 s | 0.23 s | 67 ms | 943 ms | 0 |
| 16 | 211.9 | 29.4 | 0.89 s | 9.78 s | 69 ms | 864 ms | 0 |
| 32 | 213.4 | 28.6 | 0.75 s | 28.68 s | 69 ms | 880 ms | 0 |
| 64 | 206.0 | 27.5 | 0.61 s | 68.01 s | 70 ms | 852 ms | 0 |

### llama.cpp, 12 slots

The same, with `-c 12288 --parallel 12 --n-predict 512`, the most slots that fit in `--vram-budget 22826M`:

| Requests | Tokens/s | Tokens/s per request | Time to first token | Wait | Median between tokens | Longest gap | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 36.1 | 46.8 | 3.24 s | 0.01 s | 46 ms | 299 ms | 0 |
| 2 | 67.3 | 34.9 | 0.29 s | 0.07 s | 65 ms | 206 ms | 0 |
| 4 | 153.0 | 44.0 | 0.71 s | 0.09 s | 49 ms | 456 ms | 0 |
| 8 | 204.4 | 33.8 | 2.81 s | 1.47 s | 64 ms | 220 ms | 0 |
| 16 | 135.8 | 21.4 | 1.81 s | 5.43 s | 86 ms | 1201 ms | 0 |
| 32 | 197.5 | 20.9 | 1.17 s | 21.04 s | 87 ms | 1220 ms | 0 |
| 64 | 206.0 | 20.3 | 0.85 s | 56.35 s | 90 ms | 1148 ms | 0 |

### Comparison

| Requests | llama.cpp 8 slots tokens/s | llama.cpp 12 slots tokens/s | inmoloc tokens/s | llama.cpp 8 slots time | llama.cpp 12 slots time | inmoloc time |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 34.5 | 36.1 | 46.4 | 14.8 s | 14.2 s | 11.0 s |
| 2 | 64.6 | 67.3 | 69.5 | 15.8 s | 15.2 s | 14.7 s |
| 4 | 150.5 | 153.0 | 173.0 | 13.6 s | 13.4 s | 11.8 s |
| 8 | 216.6 | 204.4 | 273.2 | 18.9 s | 20.0 s | 15.0 s |
| 16 | 211.9 | 135.8 | 382.3 | 38.7 s | 60.3 s | 21.4 s |
| 32 | 213.4 | 197.5 | 369.7 | 76.8 s | 83.0 s | 44.3 s |
| 64 | 206.0 | 206.0 | 386.6 | 159.1 s | 159.0 s | 84.8 s |

### Mixed requests

On an earlier build, on the same GPU and settings, N requests of 64 to 2048 input tokens and 32 to 512 output tokens were sent at once, one run per level:

| Requests | Tokens/s | Seconds per request | Median between tokens | Failed |
| --- | --- | --- | --- | --- |
| 1 | 31.1 | 2.20 | 50 ms | 0 |
| 2 | 37.4 | 2.80 | 69 ms | 0 |
| 4 | 81.7 | 3.05 | 51 ms | 0 |
| 8 | 103.6 | 2.36 | 62 ms | 0 |
| 16 | 129.8 | 2.15 | 83 ms | 0 |
| 32 | 178.9 | 1.79 | 125 ms | 0 |
| 64 | 155.7 | 1.75 | 126 ms | 0 |

### Serving

On one RTX 5090 laptop GPU (24 GiB) with Qwen3.8-27B IQ4_XS, MTP speculative decoding, the vision encoder, a q8_0 cache of 131072 tokens and `--vram-budget 22G`:

- **14 requests at once**, and serving in about 5 seconds from start.
- **No memory growth while serving**: device memory stayed within 6 MiB over 104 OCR requests, 16 at a time.
- **Long prompts next to other requests**: two 56k-token documents were processed at 680 to 730 tokens per second each, with short requests generating alongside.
- **q8_0 cache quality**: at 32k, 64k and 110k tokens of context, the q8_0 cache answered retrieval questions at 10%, 50% and 90% depth exactly as the f16 cache did, token for token (`scripts/kv-quality.py`).
- **CPU temperature with `--gpu-wait block`**: on an MSI Vector 16 HX AI, the CPU averaged below 65 degC while serving, against about 100 degC with the driver's default.

### OCR throughput against llama.cpp at the same memory

Both servers ran alone on the same RTX 5090 laptop GPU, one at a time, with the same model and settings:

- **Model**: PaddleOCR-VL 1.6, Q8_0 weights with its vision encoder (mmproj).
- **Common settings**: bf16 KV cache (`-ctk bf16 -ctv bf16`), `--flash-attn on`, all layers on the GPU, no prompt cache in host memory (`--cache-ram 0`).
- **Memory**: about 5 GiB of device memory each. llama.cpp (b11804, the `server-cuda` image of 2026-10-22) has no memory budget, so its cache was set by hand to the largest that fits: 45 slots of 4096 tokens (`-c 184320 --parallel 45`). inmoloc was given `--vram-budget 5G` and sized its cache and concurrency itself.
- **Requests**: pages of scanned documents rendered at 300 dpi, about 1,250 image tokens each, sent to `/v1/chat/completions` with the prompt `OCR:`, temperature 0 and at most 1024 output tokens. At N requests at once, N clients send 3N requests in total, each client one request after another.
- **Tokens/s** counts the output tokens of all requests over the wall time of the level.

| Requests at once | llama.cpp tokens/s | inmoloc tokens/s | llama.cpp median latency | inmoloc median latency |
| --- | --- | --- | --- | --- |
| 1 | 421 | 400 | 0.8 s | 0.8 s |
| 8 | 741 | 938 | 3.5 s | 3.2 s |
| 32 | 751 | 1,237 | 14.7 s | 9.3 s |
| 64 | 750 | 1,047 | 31.7 s | 24.9 s |
| 128 | 758 | 1,058 | 72.5 s | 42.4 s |
| 256 | 767 | 1,065 | 158.5 s | 49.3 s |

- From 64 requests at once, the vision encoder is the limit: every page is encoded once, one at a time, and the encoder was busy 90 to 95% of the time, at about 0.45 s per page while sharing the GPU with the generating requests (about 0.19 s with the GPU to itself). More requests at once then wait for the encoder without adding throughput. Encoding first and generating after, or giving the encoder the higher GPU priority, measured slower, as the GPU is already busy with both. Where this limit sets in depends on the GPU and the vision model.
- Every level starts from the first page, so a level that runs after lower ones on the same server can find some of its pages still cached. The inmoloc rows for 8 and 32 ran after the lower levels, and 32 includes such reuse; 64 and 128 each ran on a fresh server; 256 ran after 1 and 128, when the cache was full of running requests and almost none of its pages were still cached.
- With 1 request at a time, inmoloc's throughput is about 5% lower than llama.cpp's: each generated token takes 1.45 ms against 1.20 ms, while the prompts take the same time.

With a prompt cache in host memory on both servers (`--cache-ram 4096`), each running the levels 1 to 256 one after another on a fresh server, the pages seen in a lower level can be served from a cached prompt without encoding the image again:

| Requests at once | llama.cpp tokens/s | inmoloc tokens/s | llama.cpp median latency | inmoloc median latency | Requests from a cached prompt, llama.cpp | inmoloc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 431 | 406 | 0.8 s | 0.8 s | 2 of 8 | 2 of 8 |
| 8 | 722 | 975 | 3.7 s | 3.3 s | 8 of 24 | 8 of 24 |
| 32 | 726 | 1,277 | 15.0 s | 8.9 s | 24 of 96 | 24 of 96 |
| 64 | 1,030 | 1,504 | 25.0 s | 16.1 s | 96 of 192 | 96 of 192 |
| 128 | 823 | 1,025 | 65.6 s | 41.7 s | 60 of 384 | 119 of 384 |
| 256 | 788 | 924 | 154.5 s | 53.4 s | 0 of 768 | 6 of 768 |

At 128 requests at once, inmoloc served twice as many requests from a cached prompt: the prompts of finished requests stay in device memory, shared across requests, until their cells are needed, and then go to the host cache. At 256, the pages are new to both servers.

## Compared with llama.cpp and vLLM

llama.cpp at commit 56381e407 (2026-09-12); vLLM as of 2026-10-15.

| | llama.cpp | vLLM | inmoloc |
| --- | --- | --- | --- |
| Startup | Loads the model | Profiles memory, captures CUDA graphs, compiles the model | Loads the model; with elastic memory, no dry runs |
| GPU memory taken | What the model and context need | A fixed fraction (`--gpu-memory-utilization`) | What the model and context need, within `--vram-budget` |
| KV cache | `--parallel` fixed parts by default, shared with `--kv-unified` | Shared pool of blocks | Shared, per token |
| Requests at once | `--parallel`, at most 256 | A configured limit | Elastic memory: what fits, with no fixed limit |
| Memory per request | Fixed at startup | Fixed at startup | Taken at admission with elastic memory, else fixed at startup |
| Cache full | Every active request fails | Newest requests preempted | Elastic memory: requests wait, running ones never run out. Other models: newest requests preempted |
| Waiting requests | Wait for a slot; waiting requests are offered again when one frees | Arrival-order queue | Elastic memory: no slots, a queue partitioned by cost in blocks; the earliest request that fits starts |
| Attention reads | All used cells | Own blocks | Own cells (CUDA) |
| Shared prefixes | Own slot only | Across requests | Across requests |
| Quantized KV cache | q8_0, q4_0 and others | fp8 | q8_0, q4_0 and others |
| GGUF models | Native | Experimental plugin | Native |
| Multi-GPU | Layer and row split | Tensor and pipeline parallel | Layer and row split |

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

# inmoloc

<div align="center">

<b>Concurrent LLM serving on your own GPU, with the startup time and memory footprint of a local runtime.</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

inmoloc (Inferencia de Modelos Locales) serves many requests at once from one GPU. It starts in seconds, fits the number of concurrent requests to the memory you give it, and keeps each request's cost tied to its own context. It runs GGUF models, including vision models, through an OpenAI-compatible HTTP API.

inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends. It is maintained as its own project; see [differences from llama.cpp](docs/differences-from-llama.cpp.md) and [NOTICE](NOTICE).

## Features

- **Shared KV cache**: all requests share one cache, and each request takes cells only for the tokens it holds, up to the model's training context.
- **Elastic memory**: on hybrid gated delta-net models such as Qwen3.5, all request memory is one box of device memory, with nothing set aside per request at startup. A request's recurrent state and input are known when it arrives, so it takes them at admission; its output is charged up to `--max-output-tokens` and taken in blocks of 128 tokens as it generates. States sit at one end of the box and KV blocks at the other, and the rows of an MTP draft live in the blocks of the tokens they belong to, so any mix of short and long requests fits as long as its bytes do.
- **Memory budget**: at startup the server measures the weights, compute buffers, draft model, vision encoder, backend scratch memory, and what the process holds outside them (CUDA context, loaded kernels, libraries; measured for the process through NVML on NVIDIA GPUs). What those fixed costs leave of `--vram-budget` is request memory: the elastic memory where the model supports it, otherwise a KV cache and per-request state sized at startup. Memory does not grow while serving.
- **Admission**: with elastic memory, a request is admitted when its state, input and output charge fit. Idle slots give up their cached prompts to make room, and otherwise the request waits; waiting requests start in arrival order. A running request has all the memory it can need, so nothing is preempted.
- **Preemption on other models**: a request is admitted when its prompt and the output it is expected to generate (the mean of recent requests, at most its limit) fit in free memory. When running requests still outgrow the cache, the newest request is paused and the others continue. A paused request's state goes to the prompt cache in host memory (`--cache-ram`) and is restored when it resumes; without room there, it is recomputed.
- **Per-request attention** (CUDA): each request reads only its own KV cells, so its cost grows with its own context, not with other requests. Quantized q8_0 and q4_0 caches are read directly.
- **Prefix sharing**: a new request starts from the cached cells of another request with the same prefix, without copying them.
- **Asynchronous vision encoder**: images are encoded on their own thread while other requests generate.
- **Replay rollback for speculative decoding**: on hybrid gated delta-net models such as Qwen3.5, memory per request stays nearly the same at any draft depth.
- **Low CPU load while waiting for the GPU** (CUDA): with `--gpu-wait block` the server thread sleeps instead of spinning a CPU core.
- **Observability**: the startup log shows the memory projection and what fills it; Prometheus metrics cover KV cells, why requests wait before they get a slot, preemptions and the tokens restored after them, prefix sharing, attention cost and free device memory; `--slow-loop-ms` logs slow server iterations.

## Quick start

Run the published CUDA image (built with CUDA 12.8, so it needs NVIDIA driver 570 or newer, like llama.cpp's; see [Docker](docs/docker.md)):

```sh
docker run --gpus all -p 8080:8080 -v /path/to/models:/models ghcr.io/proafxin/inmoloc:latest \
    -m /models/model.gguf -ngl 999 --flash-attn on --ctx-size 131072 --vram-budget 22G --parallel 64
```

Or build it yourself, for any hardware llama.cpp supports (see [the build guide](docs/build.md) for other backends and older CUDA versions):

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

The log states what the budget holds, for example:

```text
limited by the budget: 14 sequences, 131072 tokens of context: model 13061 + context 6763 + compute 297 + draft 725 + other 1662 = 22508 MiB of 22528 MiB
```

A model with elastic memory needs the most tokens a request may generate, and no context size. With MTP drafting it also needs replay rollback:

```sh
build/bin/local-inference-server -m qwen3.8.gguf -ngl 999 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --rs-rollback replay --max-output-tokens 512 --vram-budget 22G --port 8100
```

```text
the elastic memory takes 5675 MiB: model 16053 + compute 529 + draft 278 + other 0 + runtime 290 are the fixed costs of the budget of 22826 MiB
```

## Server options

The options below are specific to inmoloc; all other options are as in llama.cpp, see [the server documentation](tools/server/README.md).

| Option | Default | Description |
| --- | --- | --- |
| `--vram-budget SIZE` | free device memory | Device memory the server may use, summed over the devices the model runs on (e.g. `22G`). |
| `--max-output-tokens N` | required with elastic memory | The most tokens a request generates; its `max_tokens` is capped at it, and this much output is charged when it is admitted. |
| `--ctx-size N` | automatic | Without elastic memory, the most tokens the KV cache holds; without it the cache is sized from the budget. Not used with elastic memory. |
| `--parallel N` | auto | Upper limit on requests at once. With elastic memory it only limits admission and takes no memory; otherwise the budget may lower it. |
| `--prefix-share`, `--no-prefix-share` | enabled | Let a new request start from another request's cached prefix. |
| `--prefix-share-min N` | `256` | Least number of tokens a shared prefix must save. |
| `--rs-rollback {snapshot,replay}` | `snapshot` | How recurrent state rolls back rejected speculative tokens. `replay` applies to gated delta-net models, and elastic memory with speculative drafting needs it. |
| `--gpu-wait {spin,yield,block}` | driver | How a thread waits for the GPU (CUDA). `block` sleeps and frees the CPU core. |
| `--slow-loop-ms N` | `1000` | Log a time breakdown of server iterations slower than N ms. `0`: off. |

When `--gpu-layers` is not given and the model does not fit, `--fit` chooses it against the budget, with no margin.

## Measured

### Elastic memory

On one RTX 5090 laptop GPU (24 GiB) with Qwen3.8-27B, MTP drafting of 2 tokens, replay rollback, a q8_0 cache and `--vram-budget 22826M`, 128 clients sent requests of 64 to 2048 input tokens and 32 to 512 output tokens:

- **Memory**: a box of 5675 MiB, with a state of 172 MiB and 4.8 MiB per block of 128 tokens. The draft takes 278 MiB of fixed memory, and its tokens live in the blocks.
- **Requests at once**: 20 on average, set by what the requests take and not by a number chosen at startup.
- **Throughput**: 159 tokens/s, no failed requests, a median of 128 ms between tokens.

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
| Startup | Loads the model | Profiles memory, captures CUDA graphs, compiles the model | Loads the model after short memory dry runs |
| GPU memory taken | What the model and context need | A fixed fraction (`--gpu-memory-utilization`) | What the model and context need, within `--vram-budget` |
| KV cache | `--parallel` fixed parts by default, shared with `--kv-unified` | Shared pool of blocks | Shared, per token |
| Memory per request | Fixed at startup | Fixed at startup | Taken at admission with elastic memory, else fixed at startup |
| Cache full | Every active request fails | Newest requests preempted | Elastic memory: requests wait, running ones never run out. Other models: newest requests preempted |
| Attention reads | All used cells | Own blocks | Own cells (CUDA) |
| Shared prefixes | Own slot only | Across requests | Across requests |
| Quantized KV cache | q8_0, q4_0 and others | fp8 | q8_0, q4_0 and others |
| GGUF models | Native | Experimental plugin | Native |
| Multi-GPU | Layer and row split | Tensor and pipeline parallel | Layer and row split |

## Compatibility

- **Models**: GGUF models supported by llama.cpp at the base commit, see [docs/models.md](docs/models.md).
- **API**: OpenAI-compatible chat completions, completions and embeddings, see [the server documentation](tools/server/README.md).
- **Hardware and backends**: the same as llama.cpp: all its backends build and run, on the same GPUs and CPUs. The memory budget, preemption, prefix sharing and prompt scheduling work on every backend. Elastic memory needs CUDA on Turing or newer, flash attention and a KV cache whose K and V rows have one size; elsewhere the server sizes a KV cache and per-request state at startup. Per-request attention and `--gpu-wait` are NVIDIA only (per-request attention for prompt chunks needs Turing or newer); elsewhere attention reads the whole cache with a mask, as in llama.cpp.
- **Multiple GPUs**: llama.cpp's multi-GPU modes work (see [Multi-GPU](docs/multi-gpu.md)); the budget is summed over devices, and multi-GPU serving is not tested yet.

## Coming from llama.cpp

Most `llama-server` command lines run unchanged under the new name. To let inmoloc size the server for you, replace the fixed sizes with a memory budget:

```sh
# llama.cpp
llama-server -m model.gguf -ngl 999 --flash-attn on --ctx-size 65536 --parallel 16 --kv-unified

# inmoloc
local-inference-server -m model.gguf -ngl 999 --flash-attn on --vram-budget 22G
```

| llama.cpp | inmoloc |
| --- | --- |
| `llama-server`, `llama-cli`, ... | `local-inference-server`, `local-inference-cli`, ...; unified binary `local-inference` |
| `--ctx-size N`: the KV cache size | The most the KV cache holds. Leave it out to size the cache from `--vram-budget`. Not used with elastic memory |
| `--parallel N`: the number of slots | An upper limit; the budget runs as many requests at once as fit |
| `--kv-unified-per-slot N` | `--vram-budget` sizes the cache, `--ctx-size` caps it |
| `--kv-unified` | Always on; `--no-kv-unified` is refused |
| `--fit-target` | Not used by the server: `--vram-budget` is the limit, with no margin |
| `LLAMA_ARG_*` environment variables | `LOCAL_INFERENCE_ARG_*` |
| `llamacpp:*` metrics | `local_inference:*` |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` | `ghcr.io/proafxin/inmoloc:latest` |

The HTTP API, GGUF models and chat templates are the same. The C API is source compatible; bindings that copy the `llama_context_params` layout need updating. All option changes are listed in [differences from llama.cpp](docs/differences-from-llama.cpp.md#options).

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

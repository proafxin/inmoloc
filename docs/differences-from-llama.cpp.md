# Differences from llama.cpp

inmoloc started from llama.cpp at commit 56381e407 (2026-09-12) and is maintained as its own project since. This page lists where the two differ and why. The comparison is with llama.cpp as of that commit.

## Goal

| | llama.cpp | inmoloc |
| --- | --- | --- |
| Target | Local inference on a very wide range of hardware | Serving many requests at once from one GPU, in production |
| Design priority | Simplicity and uniform behavior across all backends | Serving behavior: concurrency, latency under load, predictable memory |
| Serving paths | The same on every backend | CUDA first; other backends build and run, and fall back to the general paths |
| Direction | Upstream project with its own contribution policy | Own direction; model support is ported from llama.cpp where needed |

## Serving

| Area | llama.cpp | inmoloc |
| --- | --- | --- |
| Number of requests at once | Fixed by `--parallel` | The most that fit the memory budget (`--vram-budget`), up to `--parallel` |
| KV cache per request | With `--parallel N`, by default the context is split into N fixed parts; with `--kv-unified`, one cache shared by all requests | Always one cache shared by all requests; a request takes cells for the tokens it holds |
| Memory projection | Fits the model and context to free memory with a margin (`--fit`) | Measures every part (weights, KV cache, per-request state, compute, draft, vision encoder, backend scratch) with no margin, and sizes both the KV cache and the number of requests from the typical request size (`--ctx-typical`) |
| Memory outside the buffers (CUDA pools, cuBLAS) | Grows while serving, when an operation first needs it | Sized from the worst-case graphs and reserved at startup |
| KV cache full | Retries with smaller batches; when one token does not fit, every active request fails ("Context size has been exceeded.") and its cache is cleared | The newest requests give their cells back and are recomputed later (vLLM-style preemption); the others continue |
| Prompt and generation in one batch | Prompt tokens fill the batch next to the generated tokens | Prompt tokens are admitted by their measured cost, which grows with depth (`--prompt-cap`) |
| Shared prefixes | A request reuses the prefix its own slot holds | A request can also start from the cells of another request with the same prefix (`--prefix-share`) |
| Images | Encoded on the server loop, while generation waits | Encoded on their own thread and GPU stream at a lower priority, while the other requests generate |
| Waiting for the GPU (CUDA) | Chosen by the driver, usually a busy CPU core | Also `--gpu-wait block`: the thread sleeps until the GPU is done (`spin` and `yield` are the other values) |

## Attention and KV cache

| Area | llama.cpp | inmoloc |
| --- | --- | --- |
| Cells a token attends to | All used cells of the cache, with a mask that hides other requests | Only the cells of its own request: in place through a list for prompt chunks, through indexed ranges for generated tokens (CUDA) |
| Cost of a request | Grows with the cached prompts of every other request | Grows with its own context |
| Quantized KV cache (q8_0, q4_0) with prompt batches on CUDA | The cache view is converted to f16 for each attention op | Read directly by the per-request paths, no f16 copy |
| Recurrent state rollback for speculative decoding (hybrid models) | One state per draft position | Also `--rs-rollback replay`: one state per request, the last step's inputs are replayed |

## Observability

| Area | llama.cpp | inmoloc |
| --- | --- | --- |
| Startup memory | Buffer sizes | The projection, what fills it, and the device memory in use after each stage |
| Metrics | Throughput and request counts | Also preemptions, KV cells read against cells owned, idle cached tokens, lowest free device memory |
| Stalls | - | `--slow-loop-ms` logs a time breakdown of slow server iterations |

## Options

Options of `local-inference-server` that are new, or that mean something else than in `llama-server`. All other options are the same.

### New

| Option | Default | Description |
| --- | --- | --- |
| `--vram-budget SIZE` | free device memory | Device memory the server may use, summed over the devices the model runs on |
| `--ctx-typical N` | the longest request that fits | Tokens a typical request holds; sets the number of requests at once and the KV cache size |
| `--prompt-cap N` | `-1` (one ubatch) | Prompt tokens per iteration while requests generate, weighted by their cost; `0` = no cap |
| `--prefix-share`, `--no-prefix-share` | enabled | Let a new request start from another request's cached prefix |
| `--prefix-share-min N` | `256` | Least number of tokens a shared prefix must save |
| `--rs-rollback {snapshot,replay}` | `snapshot` | How recurrent state rolls back rejected speculative tokens |
| `--gpu-wait {spin,yield,block}` | driver | How a thread waits for the GPU (CUDA) |
| `--slow-loop-ms N` | `1000` | Log a time breakdown of server iterations slower than N ms; `0` = off |

### Changed

| Option | llama.cpp | inmoloc |
| --- | --- | --- |
| `-c`, `--ctx-size` | The KV cache size; `0` = the training context | The most the KV cache holds; not given, the cache is sized from `--vram-budget` and `--ctx-typical` |
| `-np`, `--parallel` | Number of slots; auto = 4 | Upper limit on requests at once; the budget may lower it; auto = as many as fit |
| `--fit`, `--fit-target` | Chooses the context size and GPU layers, leaving `--fit-target` (1 GiB) free per device | Chooses only the GPU layers, against `--vram-budget`, with no margin; `--fit-target` is not used by the server |
| `-kvu`, `--kv-unified` | Off with an explicit `--parallel`, the context is split into fixed parts per slot | Always on; `--no-kv-unified` is refused |
| `--cache-idle-slots` | Saves idle slots to the prompt cache when a new task starts | Saves an idle slot to the prompt cache when its memory is needed; until then it stays in the KV cache |

### Removed

| Option | Instead |
| --- | --- |
| `--kv-unified-per-slot N` | `--ctx-typical N` sizes the cache for requests of N tokens, and `--ctx-size` caps it |

Environment variables start with `LOCAL_INFERENCE_` instead of `LLAMA_` (e.g. `LOCAL_INFERENCE_ARG_CTX_SIZE`).

## What stays the same

- The GGUF format and the models it supports.
- The ggml library and the `llama_*` C API (`include/llama.h`), so code written against llama.cpp's library builds against inmoloc's. The API has additions, and `llama_context_params` has new fields, so bindings that copy the struct layout (e.g. through ctypes) need updating.
- Quantization types, backends and the OpenAI-compatible HTTP API.

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
| Memory projection | Fits the model and context to free memory with a margin (`--fit`) | Measures every part (weights, KV cache, per-request state, compute, draft, vision encoder, backend scratch) with no margin |
| Memory outside the buffers (CUDA pools, cuBLAS) | Grows while serving, when an operation first needs it | Sized from the worst-case graphs and reserved at startup |
| KV cache full | No preemption | The newest requests give their cells back and are recomputed later (vLLM-style preemption) |
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

## What stays the same

- The GGUF format and the models it supports.
- The ggml library and the `llama_*` C API (`include/llama.h`), so code written against llama.cpp's library works with inmoloc's.
- Quantization types, backends and the OpenAI-compatible HTTP API.

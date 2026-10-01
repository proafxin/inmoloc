# inmoloc

<div align="center">

<b>Concurrent LLM serving on your own GPU, with the startup time and memory footprint of a local runtime.</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

inmoloc (Inferencia de Modelos Locales) serves many requests at once from one GPU. It starts in seconds, fits the number of concurrent requests to the memory you give it, and keeps each request's cost tied to its own context rather than to everything else in the cache. It runs GGUF models through an OpenAI-compatible HTTP API, including vision models.
 
inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends. See [NOTICE](NOTICE) for credits.

## Why a separate project

llama.cpp is built for running models locally on a very wide range of hardware, and deliberately keeps its code as simple as possible so that a small team can maintain it across all of that hardware. It serves one user very well, and several at once through a basic scheduler.

inmoloc has a different goal: serving many requests at once from a single GPU in production, as vLLM does, without vLLM's long startup and large memory reservations. Getting there took changes that cut across the server, the KV cache, the compute graph and the CUDA kernels at once: memory accounting that sizes concurrency at startup, preemption, attention that reads only a request's own cells, and a scheduler that weighs prompt tokens by their cost. These trade simplicity and backend uniformity for serving behavior, which is a different balance than llama.cpp keeps, and too invasive to fit its scope as contributions.

So inmoloc is maintained as its own project with its own direction, rather than as a fork that follows upstream. It keeps llama.cpp's history, its C API and the GGUF format, and ports new model support from llama.cpp where it is needed. [docs/differences-from-llama.cpp.md](docs/differences-from-llama.cpp.md) lists the differences in detail.

## What inmoloc adds

### Memory

- **Concurrency sized to a memory budget** (`--vram-budget`): at startup the server measures the weights, the vision encoder, the draft model, a KV cache of `--ctx-size` tokens shared by all requests, the per-request state and the worst-case compute, and runs the most requests at once that fit, up to `--parallel`.
- **A complete projection, with no safety margin**: memory the backends take outside their buffers (CUDA memory pools, cuBLAS workspaces) is sized from the worst-case graphs and reserved at startup, so the server does not grow while it runs.
- **Recurrent state rollback by replay** (`--rs-rollback replay`): hybrid models such as Qwen3.5 keep one recurrent state per request and replay the inputs of the last step to undo rejected speculative tokens, instead of keeping one state per draft position.

### Scheduling

- **Preemption and resume**: when the shared KV cache runs out of cells, the newest requests give theirs back and are recomputed once there is room, so older requests keep going (the scheme of vLLM, for a unified cache).
- **A cost-based prompt cap** (`--prompt-cap`): prompt tokens are admitted per iteration by their measured cost, which grows with depth, so generating requests are not held up by a long prompt next to them.
- **Prefix sharing across requests** (`--prefix-share`): a new request starts from the cells of another request that holds the same prefix.
- **An asynchronous vision encoder**: images are encoded on their own thread and GPU stream, at a lower stream priority than text generation, while the other requests keep generating.

### Attention

- **Each request reads only its own KV cells**: prompt chunks read the cells of their sequence in place through a list of them, and generated tokens read theirs through indexed ranges, instead of the whole used cache with a mask. The cost of a request no longer grows with the cached prompts of other requests.
- **The same for quantized KV caches**: q8_0 and q4_0 caches (also f16 and bf16) use these paths directly, without an f16 copy of the cache.

### Observability

- Startup logs the memory projection, what fills it, and the device memory in use after each stage.
- Prometheus metrics for preemptions, the KV cells attention reads against the cells requests own, idle cached tokens and the lowest free device memory.
- `--slow-loop-ms` logs a time breakdown of slow server iterations.

## Quick start

Build with CUDA (see [the build guide](docs/build.md) for other backends):

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target local-inference-server
```

Serve a model with a 128k-token context shared by all requests, a q8_0 KV cache and a 22 GiB memory budget:

```sh
build/bin/local-inference-server -m model.gguf --mmproj mmproj.gguf -ngl 999 --flash-attn on \
    --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 \
    --vram-budget 22G --parallel 64 --metrics --port 8100
```

The log states how many requests run at once and what the budget holds, for example:

```text
limited by the budget: 14 sequences, 131072 tokens of context: model 13061 + context 6763 + compute 297 + draft 725 + other 1662 = 22508 MiB of 22528 MiB
```

## Measured

On one RTX 5090 laptop GPU (24 GiB) with Qwen3.8-27B IQ4_XS, MTP speculative decoding, the vision encoder, a q8_0 cache of 131072 tokens and `--vram-budget 22G`:

- **14 requests at once**, and serving in about 5 seconds from start.
- **No memory growth while serving**: the server's device memory stayed within 6 MiB over a run of 104 OCR requests, 16 at a time.
- **Long prompts next to other requests**: two 56k-token documents were processed at 680 to 730 tokens per second each, with short requests generating alongside.
- **q8_0 cache quality**: at 32k, 64k and 110k tokens of context, the q8_0 cache answered retrieval questions at 10%, 50% and 90% depth exactly as the f16 cache did, token for token (`scripts/kv-quality.py`).

## Compatibility

- Models: GGUF models supported by llama.cpp at the base commit (see [docs/models.md](docs/models.md)).
- API: OpenAI-compatible chat completions, completions and embeddings, see [the server documentation](tools/server/README.md).
- Backends: all backends of llama.cpp build and run. The per-request attention paths are implemented for CUDA; other backends fall back to attention over the cache with a mask.

## Documentation

- [Server](tools/server/README.md)
- [Build](docs/build.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Models](docs/models.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

MIT, see [LICENSE](LICENSE). inmoloc is based on llama.cpp by the ggml authors; see [NOTICE](NOTICE). Third-party components keep their own licenses, see [licenses/](licenses/) and [vendor/](vendor/).

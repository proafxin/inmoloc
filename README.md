# inmoloc

`inmoloc` is an inference engine that allows requests of arbitrary context windows elastically at the maximum possible concurrency not exceeding a specified VRAM budget and provides significantly superior scalability to llama.cpp or vllm.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

inmoloc is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and keeps its model support, quantization formats and backends (see [NOTICE](NOTICE)).

## Why

`llama.cpp` has a wide range of compatibility but a total context window and number of slots had to be chosen manually. Similarly, vllm would require manually choosing context windows and would provide better request management and throughput but would make the sacrifice of much higher startup time and additional vram costs. While llama.cpp had unified kv cache it failed requests once cache was full. I then started thinking if we could have the best of both worlds without the drawbacks that incur extra costs. `inmoloc` has evolved significantly since then. As a result, inmoloc retains the superior wider support range (making a small concession for now that the model has to be fully loaded in one gpu) of `llama.cpp`, achieves maximum possible concurrency without adding any substantial overhead like `vllm`.

I had to make `inmoloc` its own project rather than changes on top of `llama.cpp` for a number of reasons. The two have different directions and my vision fundamentally diverges from llama.cpp which aims at local inference on a very wide range of hardware even at the cost of much higher throughput that could be achieved. `inmoloc` aims at utilizing the maximum potential of the resources allocated with the only tradeoff that for now it loads the full model on a cuda GPU which is the most dominating case anyway. Its elastic design changes the attention kernels, how the KV cache and the state of each request are laid out, how the server admits requests and which options exist, so much of inmoloc differs from llama.cpp outright. Forcing that into llama.cpp, given its contribution policy (see its AGENTS.md), would make every change a maintenance burden, and its long review process would hold back the architecture change and every feature after it.

| | llama.cpp | vLLM | inmoloc |
| --- | --- | --- | --- |
| Tuning | Context size and number of slots, by trial and error | Memory fraction, context length and sequence limit, by hand | VRAM budget and the unavoidable maximum output token, nothing else |
| Wrong guess | Too large fails at load, too small wastes memory | Fails at load or leaves memory unused | Automatically sized so nothing to guess |
| Memory taken | The whole KV cache of every slot, at startup | A fixed share of the GPU, before serving | Only what running requests hold; the rest stays free for new requests and other services |
| Memory while serving | Fitted with a margin, CUDA pools grow under load | - | Measured with no margin, does not grow |
| Startup | Seconds | Profiles, captures graphs and compiles the model first | Seconds |
| Requests at once | Fixed number of slots, at most 256 | Configured limit | As many as fit, more with a larger budget |
| Context of one request | A fixed share per slot: 1024 tokens with 8 slots in 23G | Up to the configured context length | Up to ~164k tokens in the same 23G |
| Memory runs out | Every active request fails | Newest requests are preempted and recomputed later | A running request never runs out; new ones wait |
| Waiting | Constant retry, burns CPU | Arrival order: small requests wait behind large ones | The earliest request that fits starts |
| Attention cost | Every request pays for the cached context of all others | Its own blocks | Its own context only |
| Shared prefixes | Within a slot only | Across requests | Across requests |
| MTP drafting | Every slot enlarges the shared context window, more VRAM | Heavy VRAM cost: a 4-bit Qwen3.8 27B with MTP does not start on a 24 GiB GPU | Draft depth takes little extra memory |
| Recurrent state | One state per draft position, for every slot | - | Nearly the same at any draft depth (hybrid gated delta-net models such as Qwen3.5) |
| Images | Encoded while generation waits | - | Encoded while other requests generate |
| Waiting for the GPU | Usually a busy CPU core | - | Can sleep and free the core (CUDA) |
| Visibility | Buffer sizes, throughput, request counts | - | What fills the budget, why requests wait, which iterations are slow |
| **Models** | All GGUF models | GGUF only through a plugin vLLM calls [highly experimental and under-optimized](https://docs.vllm.ai/en/latest/features/quantization/gguf/) | All llama.cpp GGUF models |

`-`: not compared.

## Comparison

Setup: same RTX 5090 Laptop at different settings, `Kvark-Sparkle-1.0-27B-Q4_K_M` model, MTP draft depth 2, `q8` KV cache.

### 18G budget per server

64-token prompts, 512 output tokens per request. inmoloc: `--vram-budget 18G --max-output-tokens 512`; llama.cpp: `-c 4096 --parallel 6`.

| Requests | inmoloc wall s | inmoloc agg tok/s | llama.cpp `-c 4096 --parallel 6` wall s | llama.cpp agg tok/s | Failed (both) |
| --- | --- | --- | --- | --- | --- |
| 1 | 11.3 | 45.4 | 12.2 | 42.0 | 0 |
| 2 | 9.6 | 106.9 | 22.3 | 46.0 | 0 |
| 4 | 11.7 | 175.6 | 12.2 | 167.8 | 0 |
| 8 | 23.0 | 178.2 | 37.2 | 110.1 | 0 |
| 16 | 39.9 | 205.4 | 52.0 | 157.6 | 0 |
| 32 | 77.8 | 210.6 | 100.1 | 163.7 | 0 |
| 64 | 151.9 | 215.8 | 185.9 | 176.2 | 0 |

### 23G budget per server

128-token prompts, 512 output tokens per request. inmoloc: `--vram-budget 23G --max-output-tokens 512`; llama.cpp: `-c 8192 --parallel 8`, its largest setting that loads.

| Requests | inmoloc wall s | inmoloc agg tok/s | llama.cpp `-c 8192 --parallel 8` wall s | llama.cpp agg tok/s | Failed (both) |
| --- | --- | --- | --- | --- | --- |
| 1 | 11.3 | 45.4 | 12.8 | 39.9 | 0 |
| 2 | 9.9 | 103.0 | 9.8 | 104.1 | 0 |
| 4 | 11.9 | 172.7 | 12.8 | 160.6 | 0 |
| 8 | 15.4 | 266.4 | 18.8 | 217.7 | 0 |
| 16 | 24.3 | 336.7 | 38.2 | 214.5 | 0 |
| 32 | 49.5 | 331.1 | 76.6 | 214.0 | 0 |
| 64 | 92.2 | 355.3 | 159.3 | 205.7 | 0 |
| 128 | 183.7 | 356.8 | 311.0 | 210.7 | 0 |

- At 64 requests in 23G, inmoloc reaches 355.3 tok/s against 205.7 for llama.cpp (1.7x), and finishes in 92.2 s against 159.3 s.
- From 18G to 23G at 64 requests, inmoloc goes from 215.8 to 355.3 tok/s; llama.cpp only from 176.2 to 205.7, held back by its fixed slots.
- In 23G one inmoloc request can use up to ~164k tokens, while each llama.cpp slot gets 1024.
- On the same GPU, vLLM 0.33.0 does not start with a 4-bit Qwen3.8 27B (INT4 W4A16) and MTP drafting, while inmoloc serves the same model at Q4_K_XL with MTP drafting 2 at 387 tok/s with 64 requests in a 22826 MiB budget.

### OCR vision benchmark: PaddleOCR-VL-1.6 at 300 DPI

Setup: same RTX 5090 Laptop, `PaddleOCR-VL-1.6-Q8_0` model + mmproj, 257 pages rendered at 300 DPI (322 KiB median PNG), `max_tokens 1024`, one image per request.

inmoloc: `--vram-budget 2.5G --max-output-tokens 1024`; llama.cpp: `-c 24576 --parallel 6 --cache-ram 0 --n-predict 1024`.

| Level | inmoloc wall s | inmoloc tok/s | inmoloc GPU MiB | llama.cpp wall s | llama.cpp tok/s | llama.cpp GPU MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8 | 392 | 2262 | 0.9 | 367 | 2248 |
| 2 | 4.4 | 244 | 2262 | 2.1 | 514 | 2248 |
| 4 | 2.3 | 470 | 2262 | 2.9 | 373 | 2248 |
| 8 | 5.5 | 664 | 2262 | 6.7 | 560 | 2248 |
| 16 | 11.0 | 866 | 2262 | 12.5 | 772 | 2248 |
| 32 | 20.2 | 809 | 2262 | 23.9 | 684 | 2248 |
| 64 | 40.4 | 834 | 2262 | 47.0 | 692 | 2248 |
| 128 | 86.8 | 701 | 2262 | 93.2 | 640 | 2248 |

- Peak tok/s: inmoloc ~866 at level 16 vs llama.cpp ~772 at level 16 (~12% higher)
- At level 128: inmoloc 701 tok/s / 86.8 s vs llama.cpp 640 tok/s / 93.2 s (~10% faster)
- VRAM: essentially identical (~2260 MiB both), inmoloc's vision encoder no longer pinned to GPU
- inmoloc's level-2 outlier (4.4 s vs llama.cpp 2.1 s) is warmup on the first concurrent batch; throughput rises once the encoder is warm

## Usage

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

Serve a model with a q8_0 KV cache and a 22 GiB memory budget. inmoloc needs the most tokens a request may generate, and neither a context size nor a number of requests:

```sh
build/bin/local-inference-server -m model.gguf --mmproj mmproj.gguf -ngl 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 --max-output-tokens 512 \
    --vram-budget 22G --gpu-wait block --metrics --port 8100
```

With MTP drafting on a gated delta-net model, use replay rollback. The startup log states what the requests can use and which fixed costs the budget holds:

```sh
build/bin/local-inference-server -m qwen3.8.gguf -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --rs-rollback replay --max-output-tokens 512 --vram-budget 22826M --port 8100
```

The HTTP API, GGUF models and chat templates are the same as in llama.cpp. A `llama-server` command line needs the new name and `--max-output-tokens`, and drops the fixed sizes for a memory budget:

```sh
# llama.cpp
llama-server -m model.gguf -ngl 999 --flash-attn on --ctx-size 65536 --parallel 16 --kv-unified

# inmoloc
local-inference-server -m model.gguf -ngl 999 --max-output-tokens 512 --vram-budget 22G
```

The options below are new or differ from llama.cpp; all others are as in llama.cpp, see [the server documentation](tools/server/README.md).

| Option | llama.cpp | inmoloc |
| --- | --- | --- |
| `--vram-budget SIZE` | - | Device memory the server may use, summed over the devices the model runs on (e.g. `22G`). Default: free device memory |
| `--max-output-tokens N` | - | Required: the most tokens a request generates. A request's `max_tokens` is capped at it, and a request without `max_tokens` uses it |
| `-n`, `--n-predict N` | The default output of a request, which the request can override | Removed from the server: `--max-output-tokens` takes its place |
| `-c`, `--ctx-size N` | The KV cache size | Removed from the server: `--vram-budget` sizes the memory |
| `-np`, `--parallel N` | The number of slots | Removed from the server: as many requests run at once as fit |
| `--kv-unified-per-slot N` | The cache of each slot | Removed: `--vram-budget` sizes the cache |
| `-kvu`, `--kv-unified`, `--no-kv-unified` | Whether the slots share one KV cache | Removed: all requests share one memory |
| `--context-shift` | Shifts the context when it fills | Removed from the server: a request stays within the context window |
| `--fit`, `--fit-target`, `--fit-ctx` | Fit the model and context to free memory with a margin | Removed from the server: `--vram-budget` is the limit, with no margin |
| `-ngl`, `--gpu-layers` | Auto: `--fit` chooses how many layers fit | Auto: all layers go on the device, and a model that does not fit there fails at load |
| `-fa`, `--flash-attn` | Chooses flash attention | Removed from the server: flash attention is always on |
| `-ub`, `--ubatch-size N` | The most tokens in one GPU step | Also sets the compute memory of the model and of an MTP draft, fixed costs of the budget, and not how many requests run. Default: `512` |
| `--prefix-share`, `--no-prefix-share` | - | Let a new request start from another request's cached prefix. Default: enabled |
| `--prefix-share-min N` | - | Least number of tokens a shared prefix must save. Default: `256` |
| `--rs-rollback {snapshot,replay}` | - | How recurrent state rolls back rejected speculative tokens. `replay` applies to gated delta-net models and takes less memory per request. Default: `snapshot` |
| `--gpu-wait {spin,yield,block}` | - | How a thread waits for the GPU (CUDA). `block` sleeps and frees the CPU core. Default: driver |
| `--slow-loop-ms N` | - | Log a time breakdown of server iterations slower than N ms. `0`: off. Default: `1000` |
| `--threads-http N` | Default: at least `--parallel` + 4 | Default: one less than the CPU threads, at least 5, whatever the number of requests |
| `--cache-idle-slots` | Saves idle slots to the prompt cache when a new task starts | Saves an idle slot to the prompt cache when its memory is needed; until then it stays in the KV cache |
| `-cram`, `--cache-ram N` | The prompt cache in host memory; an entry keeps all the context checkpoints of its slot | Also holds the state of preempted requests until they resume; an entry keeps only the newest checkpoint |

When a budget is more than the device has free, 128 MiB of the device stay free, since allocations fail in the last of its memory.

### Names

| llama.cpp | inmoloc |
| --- | --- |
| `llama-server`, `llama-cli`, ... | `local-inference-server`, `local-inference-cli`, ...; unified binary `local-inference` |
| `LLAMA_ARG_*` environment variables | `LOCAL_INFERENCE_ARG_*` |
| `llamacpp:*` metrics | `local_inference:*` |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` | `ghcr.io/proafxin/inmoloc:latest` |

The C API has additions, and `llama_max_parallel_sequences()` is removed since there is no fixed limit; bindings that copy the `llama_context_params` layout need updating.

## Documentation

- [Server](tools/server/README.md)
- [Build](docs/build.md)
- [Docker](docs/docker.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Models](docs/models.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

MIT, see [LICENSE](LICENSE). inmoloc is based on llama.cpp by the ggml authors; see [NOTICE](NOTICE). Third-party components keep their own licenses, see [licenses/](licenses/) and [vendor/](vendor/).

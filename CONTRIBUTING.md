# Contributing to inmoloc

Issues and pull requests are welcome at https://github.com/proafxin/inmoloc.

## Before you start

- For anything larger than a fix, open an issue first to discuss the design. Changes to the scheduler, the KV cache, the memory projection or the attention paths affect each other, and are easier to get right before code is written.
- inmoloc is built for serving many requests at once from one GPU (see [README.md](README.md) and [docs/differences-from-llama.cpp.md](docs/differences-from-llama.cpp.md)). Judge a change by what it does for concurrent serving, and by what it costs in startup time and memory.
- Model support comes from llama.cpp. To add a model that llama.cpp supports, port its commits from the `upstream` remote rather than writing it anew.

## Pull requests

- Keep a pull request to one change, and explain what it does and why in the description.
- Commit messages use the form `type(scope): summary`, e.g. `perf(server): reuse prompt cache state buffers` or `fix(cuda): ...`.
- You are responsible for every line you submit, however it was written, including with AI assistance: be ready to explain it.
- Test what you change:
  - CUDA kernels: `test-backend-ops` for the ops you touched, e.g. `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0`
  - unit tests: `ctest --test-dir build -L main`; the recurrent state rollback also with `test-recurrent-state-rollback --rs-rollback replay` (see `.github/workflows/ci.yml`)
  - server: the pytest suite in `tools/server/tests` (`./tools/server/tests/tests.sh`, see its README), which includes preemption and prefix sharing; `scripts/bench-server-concurrency.py` for throughput and latency
  - KV cache quality at long context: `scripts/kv-quality.py`, against a server with the reference cache type and one with the type under test
- A change to serving speed needs numbers: compare with and without it on the same workload, alternating runs on fresh servers, with nothing else using the GPU.

## Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: not yet applied to all of the codebase; new code should follow it.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](ggml/include/ggml.h) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

## Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: not yet applied to all of the codebase; new code should follow it.)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

## Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

## Server

For changes to the server, see the [server development documentation](tools/server/README-dev.md).

## Documentation

When you change how something is used, update its documentation in the same pull request. When you find documentation that is wrong or out of date, fix it.

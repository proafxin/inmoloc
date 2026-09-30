# Instructions for inmoloc

inmoloc serves many requests at once from one GPU, with a local runtime's startup time and memory footprint. It is based on llama.cpp and maintained as its own project (see [README.md](README.md), [docs/differences-from-llama.cpp.md](docs/differences-from-llama.cpp.md) and [CONTRIBUTING.md](CONTRIBUTING.md)).

## Working on this codebase

- Before writing code, read the relevant files and follow their patterns: your change must blend in with the code around it.
- Judge a change by what it does for concurrent serving and what it costs in startup time and memory. Avoid designs that need long warmups or large reservations up front.
- Prefer reusing existing infrastructure over adding new subsystems. If a change is large or introduces a new pattern, pause and discuss the design with the user first.
- New options are command-line parameters, not environment variables.
- Memory is accounted for, not guessed: the startup projection must cover everything the server allocates. Do not add safety margins for memory that cannot be explained; find and measure it instead.
- Explain a problem before fixing it: find the cause with measurements (logs, counters, a comparison with the base llama.cpp at commit 56381e407 in a separate worktree), then change the code.
- Claims about speed need measurements on the same workload with and without the change, alternating runs on fresh servers, with nothing else using the GPU.
- Keep the internal `llama_*` and `ggml_*` C API and the GGUF format unchanged unless the change requires it, so that model support can still be ported from llama.cpp.
- Tests: reuse the existing ones (`test-backend-ops`, `tests/test-*.cpp`, the `scripts/check-*.sh` checks) rather than adding new test files, unless the user agrees.

## Code standards

- Use ASCII only: `-` instead of an em dash, `->` instead of an arrow, `x` instead of a multiplication sign, `...` instead of an ellipsis.
- Code comments:
    - Keep them concise, usually 1-2 lines, and only where the code does not already say it
    - Do not hard-wrap them to a fixed column width
    - Use simple wording (ASD-STE100 Simplified Technical English)
    - Write the code first, then add comments only where they are needed
- Do not split a line mid-sentence to fit a fixed width.

## Commits

- Do not commit or push unless the user asks for it; the user normally commits their own work.
- Commit messages use the form `type(scope): summary`, e.g. `perf(server): reuse prompt cache state buffers`.

## Examples

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

## Useful resources

- [Build documentation](docs/build.md)
- [Server usage](tools/server/README.md) and [server development](tools/server/README-dev.md)
- [How to add a new model](docs/development/HOWTO-add-model.md) (for inmoloc, prefer porting from llama.cpp's `upstream` remote)
- [PEG parser](docs/development/parsing.md), [auto parser](docs/autoparser.md) and the [Jinja engine](common/jinja/README.md) for chat templates and output parsing
- Reusable task workflows in [skills/](skills/)

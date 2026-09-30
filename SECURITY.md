# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in inmoloc, report it privately through a [security advisory](https://github.com/proafxin/inmoloc/security/advisories/new). Do not open a public issue for it. Include a proof of concept (a script and/or files) and the commit you tested.

inmoloc is maintained on a best-effort basis. Please allow 90 days for a fix before disclosing the issue publicly.

Vulnerabilities in the third-party code under `vendor/` should be reported to those projects. Vulnerabilities in code that inmoloc shares with llama.cpp (e.g. `ggml/`, the GGUF parser) are best reported to llama.cpp as well; inmoloc will port the fix.

### Covered

- `src/`, `ggml/`, `common/`, `gguf-py/`
- `tools/server/`, excluding the web UI, features marked experimental, and features not meant for untrusted environments (router mode, MCP)

Denial-of-service bugs are looked at case by case and are not generally treated as vulnerabilities.

## Running inmoloc securely

### Untrusted models

Run models from unknown sources only in an isolated environment such as a container or a virtual machine. How much to trust a model depends on where it comes from and how you use it.

### Untrusted inputs

Models can take text, images and audio, and the libraries that decode these differ in how hardened they are. When the inputs are not trusted:

- run inference in an isolated environment
- validate and sanitize inputs before they reach the model
- keep inmoloc and its dependencies up to date
- test how the model responds to prompt injection before exposing it

### Untrusted networks

The server is meant to run behind your own access control. If it must be reachable from an untrusted network:

- do not expose the RPC backend (`ggml-rpc-server`) or the server itself directly
- put the server behind an authenticating reverse proxy, and set `--api-key`
- verify the hashes of downloaded models against known-good values
- encrypt traffic over the network

### Several tenants on one server

Requests to one server share its model, its KV cache and its GPU. inmoloc keeps the KV cells of each request separate (a request attends only to its own cells, or to a shared prefix identical to its own), but the requests still share hardware, and research has shown side channels on GPUs. Where tenants must not affect or observe each other, give each its own server and GPU, and use rate limits and monitoring to keep one tenant from starving the others.

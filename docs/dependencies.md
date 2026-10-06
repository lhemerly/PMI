# Inference and storage dependencies

PMI retains llama.cpp/ggml for optimized numerical kernels and reference decoding.
The Git submodule is pinned to
`4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`. This replaces the original
`d81aef19941e145d04f88fb180ea89a67d052ab5`, which predates recent Qwen4exp
attention/pooling correctness fixes. Historical source investigation remains in private evidence.
The tracked `patches/llama-pmi-external-tensors.patch` adds the CPU deferred-weight
loader/graph callbacks used by PMI. CMake applies it locally through
`tools/apply_llama_patch.py` before building. The script requires this exact pin,
recognizes an already-applied patch, and stops on a conflict without resetting
local edits or fetching anything. An initialized vendor checkout therefore shows
local modifications; the reproducible source is the root repository's patch,
not an unpublished vendor commit. This extension is not an upstream API.

```sh
git submodule update --init vendor/llama.cpp
```

The patch was applied to a clean archive of the exact pin, reverse-checked, and
all 13 modified tracked files and the added observer header matched the compiled checkout byte for byte. Updating
llama.cpp requires deliberately rebasing this patch and rerunning the graph and
decoder differential tests.

The Rust `runtime/` crate uses only the standard library and exposes a small C
ABI described in `include/pmi/runtime_ffi.h`. CMake invokes Cargo with
`--offline --locked`; a checked-in lockfile makes the absence of registry
dependencies explicit. Keep kernel/backend code in C++, not in this crate.

Tested toolchain: Rust/Cargo 1.95, GCC 16.2, CMake 4.4.3, Python 3.14 on Linux.
The initial integration tests used the ggml CPU backend. Subsequent native Vulkan
checks cover real-layer resident-versus-paged projections and a bounded trained
checkpoint cache-capacity invariance gate; see [native cache invariance](native-cache-invariance.md).
These scoped checks do not establish general checkpoint compatibility, model quality
or CPU/Vulkan numerical equality. The experimental CPU-compatible path still fails
the preserved full-checkpoint CPU parity diagnostic.

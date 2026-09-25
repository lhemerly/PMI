# Inference dependency

PMI uses llama.cpp for optimized kernels, model loading, and the reference decoding path. The dependency is tracked as a Git submodule at `vendor/llama.cpp` and pinned to commit `d81aef19941e145d04f88fb180ea89a67d052ab5`.

Clone with submodules enabled:

```sh
git clone --recurse-submodules https://github.com/lhemerly/PMI.git
```

To initialize it after a plain clone:

```sh
git submodule update --init --recursive
```

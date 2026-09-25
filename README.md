# Poor Man's Inference (PMI)

PMI is an experimental C++17 runtime for explicit hierarchical model-weight residency:

`NVMe -> RAM cache -> VRAM cache`

The current milestone provides a model-independent foundation and a pinned llama.cpp reference adapter:

- GGUF inspection without loading tensor payloads
- JSON tensor manifest with absolute file offsets and byte lengths
- semantic classes for always-active, routed experts, sparse lookup, and optional tensors
- packed `_exps` tensors expanded into per-expert byte ranges
- bounded byte-based LRU cache with pinning
- asynchronous file `PageProvider`
- greedy llama.cpp reference generation with generated token IDs

The custom page provider is not yet wired into llama.cpp's model graph. Reference runs currently use its normal loader and provide the correctness baseline for future paging integration.

## Build

Clone the repository with its llama.cpp dependency:

```sh
git clone --recurse-submodules https://github.com/lhemerly/PMI.git
```

Requirements: GCC 13+ and GNU Make for the standalone inspector; CMake 3.14+ to link the reference runtime.

```sh
make
make test
```

Build PMI with pinned llama.cpp (CPU backend by default):

```sh
cmake -S . -B build -DPMI_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pmi -j2
```

## Inspect a checkpoint

```sh
./pmi inspect model.gguf > manifest.json
```

The inspector reads GGUF headers, metadata, and tensor descriptors without reading the tensor data region.

## Reference decoding

```sh
./build/pmi reference model.gguf --prompt "Say hello" --max-tokens 32 --gpu-layers 0
```

The `TOKENS` line contains generated token IDs for differential tests. `pmi run` is reserved for the explicit PMI graph path and currently reports that execution integration is pending. `--gpu-layers` selects llama.cpp reference offload when built with an accelerator backend.

## Current design and limitations

`GgufReader` owns file-format parsing. `ModelManifest` describes logical tensors and per-expert slices. `PageProvider` exposes `request`, `prefetch`, `pin`, and `release`; its initial file implementation reads requested byte ranges into bounded RAM cache storage.

This workspace pins llama.cpp at `d81aef19941e145d04f88fb180ea89a67d052ab5`, which includes merged Qwen3.8-Flash-Next (`qwen4exp`) support. The official Q8 GGUF is 163 GB, so it is not included. The current reader handles one GGUF shard at a time; multi-shard manifest aggregation remains to be implemented. No model-level token equality test has been run here.

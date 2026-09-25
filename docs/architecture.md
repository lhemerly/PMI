# PMI architecture and implementation boundary

## Current implementation

1. `GgufReader` parses GGUF metadata and tensor descriptors from the file header.
2. `ModelManifest` assigns semantic classes and expands packed `_exps` tensors into individually addressable expert byte ranges.
3. `PageProvider` asynchronously reads requested byte ranges into a bounded LRU cache. The interface separates requests, prefetch, pin, and release from the cache implementation.
4. `run_reference` uses llama.cpp to produce greedy reference tokens. The llama.cpp source is pinned under `vendor/llama.cpp`.

The custom page provider currently serves tests and tools. It is not connected to the llama.cpp model graph or its GPU buffers. Reference generation uses llama.cpp's ordinary model loader.

## Memory accounting

The configured PMI cache capacity currently bounds only bytes held in the LRU cache. It does not yet enforce process RSS or VRAM usage. A runtime budget must also reserve room for graph buffers, KV/recurrent state, token embeddings, temporary activations, and transfer staging before allocating expert pools.

The official 163 GB Q8 target is split across GGUF shards. The current reader handles one GGUF file at a time; multi-shard manifest aggregation is required before inspecting the checkpoint as one model.

## Expert range calculation

For packed MoE tensors shaped like `[input, intermediate, expert_count]`, each expert occupies a contiguous slice along the outermost GGML dimension. PMI computes one two-dimensional slice's byte size from its quantization type and shape, then records:

```text
expert_offset = packed_tensor_offset + expert_id * expert_slice_bytes
```

Each manifest entry retains the logical per-expert name and original packed tensor name. The synthetic fixture verifies all four expert offsets and reads a selected slice back from disk.

## Execution integration

The model graph must expose the router-selected expert IDs before expert matrix multiplication. PMI should load only selected expert slices into bounded host storage, transfer them into bounded GPU slots, and execute the existing ggml MoE operations against remapped slot IDs. Routing IDs and weights must be captured for telemetry.

The first integration should handle one MoE layer and one token at a time. Compare selected experts, individual expert outputs, layer output, and full greedy token IDs against the pinned llama.cpp reference before enabling prefetch overlap.

## Build baseline

- C++17 and GCC 13
- CPU-only ggml backend in this workspace
- llama.cpp commit `d81aef19941e145d04f88fb180ea89a67d052ab5`
- `make test` validates the standalone GGUF parser, manifest, cache, and file-backed page provider
- CMake builds the llama.cpp reference adapter

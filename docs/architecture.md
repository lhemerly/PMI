# PMI architecture and implementation boundary

Historical report names and unl inked investigation summaries below refer to private
receipts excluded from this public source proposal. Current evidence scope is
[ native cache invariance ](native-cache-invariance.md).

Current native computation and evidence scope: [native cache invariance](native-cache-invariance.md).

PMI executes the complete Qwen `qwen4exp` text decoder through `pmi run`, with
routed expert matrices and PLE n-gram embedding rows supplied by its paging
runtime. The existing llama.cpp graph owns attention, hyper-connections, routing,
shared experts, KV/recurrent state and PLE history. `pmi reference` uses the
ordinary loader and graph operations without PMI page storage.

The current boundary supports CPU execution, one sequence, greedy generation and
MTP-disabled loading. Complete synthetic models have passed full-graph parity
tests. The pretrained, hash-verified Qwen3.8-Flash-Next checkpoint passed a
real-layer comparison and exact two-token native demand-mmap parity. A separate
bounded 16-token run completed; only its first two tokens are reference-validated.
See trained-model evidence.

## Implemented components

| Component | Responsibility |
| --- | --- |
| C++ `GgufReader` and `ModelManifest` | Parse headers, validate a complete shard set, preserve each tensor's shard and byte range, and describe packed expert slices. |
| Rust `pmi-runtime` | Own open shard files, range validation, aligned allocations, cache keys, eviction, pins, leases, and storage counters. It has no model or quantization logic and uses only the Rust standard library. |
| C ABI and C++ `FilePageProvider` | Expose Rust allocations as immutable `Page` objects and translate status failures into exceptions delivered by futures. |
| C++ `ProtectedPageRetention` | Coordinate generic opaque groups of exact ranges using existing counted pins; bound old/new turnover and release obsolete ownership after a group completes. It performs no file reads and knows no model layouts. |
| C++ `Qwen4ExpMoeConnector` | Validate Qwen tensor layouts and supported routing configuration, choose expert ranges, build CPU operations, and retain each page until its operation completes. |
| C++ `PagedQwenGraph` | Validate deferred expert/PLE descriptors, replace their graph consumers, request selected pages, and propagate callback failures. It retains stable descriptors across graph rebuilds. |
| C++ generation driver | Tokenize, prefill in chunks, decode greedily, manage the model/context lifetimes, and collect optional traces and telemetry. |
| Existing ggml/llama.cpp plus a narrow tracked patch | Supply quantized kernels and the complete Qwen graph/state machinery. The patch defers selected tensor allocation and redirects their consumers to PMI custom operations. |

The vendored backend is pinned at
`4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`. Its Qwen support and relevant fixes were
inspected in the private historical source investigation. CMake invokes
`tools/apply_llama_patch.py` to apply the checked-in
`patches/llama-pmi-external-tensors.patch` to that exact pin, without fetching.
It accepts an already-applied patch and stops on a different pin or conflicting
vendor edits. The extension is local and pin-dependent; reference mode leaves
its callbacks unset and retains the normal backend path.

## Deferred storage and full-graph execution

The loader consults PMI's selection callback before allocating tensor buffers.
Selected expert arrays and the PLE table keep their metadata in separate contexts
with no weight buffer or payload read. Paged loading disables model mmap and
upstream lazy loading; dense weights are loaded into ordinary CPU buffers.
This prevents the loader from first allocating or mapping the full deferred
arrays. The driver passes the complete canonical shard-path list to the loader,
so renamed files cannot silently resolve to different conventionally named
siblings.

Two graph hooks consume the deferred descriptors:

- Expert `MUL_MAT_ID` receives the existing F32 activations and global I32 expert
  IDs. For each token and selected slot, PMI leases one expert matrix and runs
  the retained ggml CPU matrix kernel. Gate/up/down, routing and aggregation
  retain the model graph's original ordering and semantics.
- PLE `GET_ROWS` receives the global row IDs computed by the existing n-gram hash
  and history logic. PMI leases each encoded row and copies or dequantizes it into
  the F32 activation output using ggml's type traits.

Custom nodes depend on activations and IDs, never on an unallocated deferred
weight tensor. The patch validates their operation, output type and shape before
execution. Source descriptors live independently of graph tensor pointers; the
provider and connector outlive the model and decoder context. Expert operations
currently process one matrix with one CPU task at a time. This bounds page lease
lifetimes but is not a throughput optimization.

The retained backend supplies the GDN and QSA attention paths, HC mixers, shared
experts, tokenizer, output head and state updates. PLE convolution
history and EOS resets stay in that graph rather than being reimplemented in
Rust. Prompt chunks and subsequent single-token decode calls share the same
decoder context. MTP loading is explicitly disabled.

Custom callbacks latch errors instead of throwing through C callbacks. The CPU
abort hook and checks after loading, context creation and each decode propagate
failures to the CLI. A failed run does not print successful tokens or retain a
decoder context for reuse. Recovery of a partially executed context is not an
exposed feature.

Optional `previous-top4` retention belongs to the Qwen graph connector: it chooses
the final token's top-four expert IDs, protects every gate/up/down page as it is
already requested, and retires the old selection only after down completes.
The generic retention helper owns pin bookkeeping; Rust remains the authority
for allocation, cache eviction and lifetime accounting. No new storage ABI or
kernel rewrite was needed. The planner reserves old/new turnover and service
space, reduces the quota when necessary, and preserves baseline LRU by default.
See policy details and evidence.

## Shards and page lifetime

The manifest aggregates shards by `split.no` and validates split counts, physical
tensor counts, shared metadata, duplicate names, offsets, sizes, and overlapping
physical ranges. The provider receives paths in manifest shard order. A cache
key is the exact `(shard_id, absolute_offset, byte_length)` tuple, so equal offsets
in separate shards and different lengths at one offset remain distinct.

Packed expert arrays use contiguous slices along their outermost GGML dimension:

```text
expert_offset = packed_tensor_offset + expert_id * expert_slice_bytes
```

Slice size comes from the quantization type and remaining dimensions. Entries
retain the source tensor name, expert ID, shape, and shard ID. Name-based semantic
classes and slicing conventions are advisory; the connector validates the actual
layout before execution.

`request` and `prefetch` currently perform synchronous I/O and return ready shared
futures. A cache hit shares the same Rust allocation without copying its payload.
Each returned page owns a lease that blocks eviction and remains valid after the
provider is destroyed. Pins are counted independently and must be released the
same number of times they were acquired. Shard files must remain immutable.

An insufficient-capacity request first plans all necessary evictions. If active
leases or pins prevent enough space from being reclaimed, it fails without
evicting any existing page. Invalid lengths and ranges are rejected before
allocation. I/O failure can occur after planned evictions, but an incomplete page
is never installed in the cache.

## Memory and I/O accounting

Rust aligns page storage to 64 bytes and rounds each allocation size up to 64
bytes. The cache budget and `current_bytes`/`peak_bytes` count those reserved
allocations, including alignment padding. Allocating a new page requires room
within that bound; a cache hit or additional lease creates no second payload.

This is a bound on owned page allocations, not on process RSS or VRAM. It excludes
cache and allocator metadata, connector traces, graph activations and workspaces,
model state, backend buffers, transfer staging, and the operating-system file
cache. Dense model weights remain outside the page cache. Current CPU connectors
lease at most one expert matrix or PLE row at a time; capacity must fit the largest
requested page after rounding. Metadata describing a large PLE table is retained,
but its rows are described on demand rather than expanded into millions of
manifest entries.

Runtime counters report hits, misses, evictions, failed requests, bytes reused,
active leases, pinned entries, and bytes actually returned by file reads,
including partial failed reads. File-read bytes do not imply physical disk reads.
Generation metrics separately report setup, prefill and decode timing, expert and
PLE request counts, selected deferred weight bytes, cache counters, process RSS,
peak RSS and process storage reads where available. `vram_bytes` is zero for the
CPU paged path. This does not provide a complete per-component accounting of
resident model weights, graph workspaces or decoder state. `moe-layer` also
reports layer evaluations per second; that is not token-generation throughput.

Optional `--trace-json` captures finite F32 and I32 graph snapshots, shapes,
evaluation indices, prefill/decode phase, tokens and metrics. It includes MoE,
shared expert, PLE, QSA selection and final-logit boundaries. Trace payload has a
64 MiB cap and changes scheduler observation points; tests therefore also run
ordinary generation with tracing disabled. Trace memory and timing overhead must
be excluded from production performance conclusions.

`--profile-paging on` records paged custom-operation phase timings and counts,
including separate expert and PLE request time. These exclude dense graph work
and are not a complete partition of end-to-end latency. `--routing-jsonl` captures
only routing IDs/normalized weights and aggregate/evaluation metrics, within
explicit byte/record limits; it never captures hidden activations or token IDs.
The offline SQLite dataset preserves provenance and missing
fields, with whole-run/prompt-group split labels for future causal policy tests.

## Layer and full-decoder validation

`Qwen4ExpMoeConnector` accepts one embedding after the model's HC mixer. It performs
router projection, softmax, top-k selection, normalized expert weights, selected
gate/up/down projections with SwiGLU, and the gated shared expert. It rejects
unsupported layouts and routing options. Quantization selects existing ggml CPU
kernels; it does not create a separate storage implementation.

Synthetic multishard fixtures compare selected expert IDs exactly and compare
router probabilities, weights, expert intermediates, shared output, and final
layer output against resident ggml execution. F32 additionally has a scalar
reference. Fixtures cover F32, Q8_0, IQ4_XS, IQ3_S, IQ4_NL, mixed quantization,
changed routing, warm reuse, and forced eviction/reload. These checks do not
replace the separate full-decoder tests.

`tools/make_qwen_fixture.py` builds a complete deterministic GGUF with four
layers: three GDN, one QSA, HC mixers, four experts/top-two routing, a shared
expert, PLE n-gram embeddings and a byte-fallback tokenizer. It uses only Python's
standard library and downloads no weights. `tests/full_model_test.py` compares
the ordinary reference graph against `pmi run` using those identical files.

The verified suite covers 22 differential cases plus four untraced runs, including
F32/F16/Q8_0 experts, F32/F16/IQ4_NL PLE, single and split files, arbitrary reversed
shard paths, EOS resets, and long prompts crossing QSA pool boundaries. Prefill
chunks of 1, 4 and 64 produce the same greedy tokens. The comparisons cover
2,053,178 floating-point values and 16,634 integer values with zero observed
difference; declared F32 tolerance is `2e-5 + 1e-4 * abs(reference)` and integer
values must match exactly.

Tests require actual expert and PLE requests, the expected deferred-byte total,
cache capacity compliance, warm hits, forced eviction/reload, and clean rejection
of insufficient cache or incomplete shard sets. In the F32 fixture, an 8,192-byte
cache forces repeated reloads while preserving the complete decoder output.
That number bounds cached payload only, not the process. See the
[README](../README.md) for fixture/run commands and [validation](validation.md)
for results and limitations.

## Remaining gates

The synthetic models establish full-graph wiring, state progression and storage
parity with the pinned reference. The later real-checkpoint comparisons establish
layer and two-token full-graph parity, followed by one 16-token CPU run. They do
not establish independent official-implementation correctness, reference parity
for tokens 3–16, long-context behavior or controlled cold/warm performance.
The release and optimization audit separates
these remaining gates from the implemented storage boundary and quantifies
cache-policy opportunities using the saved trace.

GPU paging, concurrent sequences, sequence-copy/rollback scenarios, MTP, vision,
asynchronous I/O overlap and production scheduling remain unimplemented or
unvalidated. The connector requires separate gate/up/down expert arrays and
rejects unsupported fused/scaled layouts or precision policies. A second model
connector and comparative performance claims should follow proof of this first
boundary with real weights. Quantization continues to select retained numerical
kernels; it does not require a separate Rust storage runtime.

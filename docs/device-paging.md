# Synchronous device paging boundary

Historical report names and unl inked investigation summaries below refer to private
receipts excluded from this public source proposal. Current evidence scope is
[ native cache invariance ](native-cache-invariance.md).

Current native computation and evidence scope: [native cache invariance](native-cache-invariance.md).

The reusable [device cache](../include/pmi/device_cache.hpp) bridges Rust-owned
host pages to backend-owned allocations. The [ggml adapter](../include/pmi/ggml_device.hpp)
uses the pinned C++ backend API. The Qwen graph now calls a model-independent
[projection executor](../include/pmi/device_projection.hpp) when explicitly
selected. The default remains its validated direct CPU path. The opt-in
`device-cpu` mode exercises the device allocation boundary on CPU.
`device-vulkan` explicitly selects a Vulkan device for routed expert projections.
Guarded projection and full tiny-Qwen GPU fixtures have passed the criteria
described below; original strict-comparison failures remain recorded separately.
Later bounded trained-checkpoint native Vulkan paging passed the scoped cache
invariance checks in [native cache invariance](native-cache-invariance.md). This
does not establish broad compatibility, quality or sustained performance.

Rust continues to own file handles, shard reads, aligned host buffers, host cache
policy, counted pins and immutable page lifetimes through the existing FFI. The
C++ device layer owns backend handles and allocation lifetimes close to ggml's
C++ implementations. There is no numerical-kernel rewrite or second file loader.
Connectors provide compact tensor descriptors; routing, global expert IDs,
selection order and scatter remain connector responsibilities.

## Admission and lifetime contract

Each cache is bound to one provider, one backend and immutable source shards.
Its key contains shard ID, absolute offset, byte length, quantization type,
shape, format and layout. Names and redundant relative offsets do not affect
identity. Format/layout tags are opaque to the core and validated by the adapter.

On a miss, the backend validates the descriptor and quotes its managed buffer
charge. The cache checks that charge and plans sufficient unleased LRU victims
before requesting host bytes or allocating a backend buffer. A failed host read
preserves existing device entries. The new host lease remains alive throughout
the synchronous upload, including cleanup on failure. A published device lease
owns its allocation and backend independently of the cache and host provider.

Consumers must retain device leases until their operations are synchronously
complete. This interface has no asynchronous fence or stream-lifetime inference.
The provider must outlive the cache, although already returned device leases may
outlive both. Cache operations are serialized; callback reentry is rejected.
`clear()` removes only unleased entries and continues charging retained entries.
Releasing a lease does not itself evict the cached allocation.

Allocation/upload failures publish no new entry and clean temporary resources.
They cannot restore previously evicted unleased victims: retaining those buffers
through a replacement allocation would exceed the strict budget. Backend
implementations must finish submitted uploads before returning or throwing.
Fatal native API failures that terminate the process are outside C++ exception
recovery.

## Budget and measurements

The device budget counts the adapter's quoted, aligned **managed buffer bytes**.
The adapter checks that the allocated buffer reports the quoted size. This
contract bounds cooperating backend allocations; it cannot prevent a faulty
backend from allocating more than its quote before the mismatch is detected.
The generic cache records requests, hits, misses, failures, successful uploads,
logical upload bytes, evictions, current/peak charges and active lease copies.
Failed uploads contribute allocation peaks but not successful upload bytes.

These counters are separate from Rust host cache bytes and physical storage
reads. They exclude metadata, activations, graph workspaces, backend-global
state, hidden staging, driver heaps and unrelated allocations. In particular,
ggml's Vulkan buffer size reports the requested suballocation, not total device
heap occupancy. Every manual GPU test here therefore requires independent
host-memory, swap, time and whole-device VRAM observations and limits.

Generation telemetry reports `vram_bytes: null` with
`vram_bytes_scope: "physical_device_usage_not_sampled"` for Vulkan. CPU paths
retain zero with `pmi_gpu_allocations_none`; that value describes PMI GPU
allocations, not other applications. Device metrics label their managed scope
`ggml_buffer_reported_allocations_excluding_driver_staging_scratch`. Neither
managed buffer total can substitute for a physical VRAM measurement.

The adapter receives an explicitly created backend owner. It never discovers or
initializes hardware. Only compact, contiguous tensors in the adapter's supported
format/layout are admitted. Caller-provided shapes are validated before native
tensor construction, including block-size and byte-length consistency. A caller
must not use this interface to pass all 512 experts when only a selected expert
matrix is needed; backend dequantization fallback can expand the entire supplied
tensor.

## Qwen integration and explicit budgets

`pmi run --projection-backend device-cpu` and `device-vulkan` require positive
`--device-cache-bytes` and `--device-graph-bytes`. The first bounds cached
backend weight allocations; the second bounds the aligned F32 input/output
buffer for one projection. Both consume host RAM in CPU-backed mode, in
addition to Rust pages and resident dense weights. Vulkan uses backend-managed
device buffers. The Rust host cache still has its separate `--cache-bytes`
budget. Neither device cap includes hidden backend workspace, driver heaps,
host vectors or metadata.

Vulkan requires a build with `PMI_WITH_LLAMA=ON`, `GGML_VULKAN=ON` and
`GGML_BACKEND_DL=OFF`. `--projection-device-index N` selects the Vulkan device;
its default is 0, and it is rejected for CPU/reference modes. CPU remains the
default and does not initialize Vulkan, even in this build. An unavailable
backend/device or unsupported operation fails explicitly, without CPU fallback.
The CLI records the selected backend name, device name/description/type, index
and optional backend-supplied `device_id`. Match that physical identity to the
VRAM guard's device; an enumeration index alone is insufficient.

The executor validates the compact descriptor, operation support and graph
buffer quote before requesting weights. It holds the weight lease, graph
buffer and readback vector through synchronized cleanup, including failures.
Only successful synchronized readback publishes an output. The Qwen connector
preserves global expert IDs, token/slot order, broadcasts and output scatter;
PLE lookup remains on the existing host path. Failure latches the outer graph's
abort, zeroes its failed output and clears cached device allocations, without
falling back to a resident full expert tensor.

The initial bridge rejects host `previous-top4` retention. It needs a separate
device pin policy; silently combining it with device hits would miss the host
pin helper's load observations. Reference commands and the default CPU path
reject device budgets. Dense weights, shared experts, routing, PLE lookup and
the outer decoder remain on CPU; `--gpu-layers` must remain 0 for paged runs.

`device_projection` metrics contain cache requests/hits/misses, successful
uploads and uploaded bytes, actual host-provider calls, graph-buffer current
and peak charges, activation/readback bytes and optional phase timings.
`cache_request_ms` includes cache admission, host I/O and weight upload;
do not interpret it as storage time alone. With profiling enabled, nested
`cache.provider_read_ms`, `allocation_ms`, `upload_ms` and `eviction_ms`
separate host-provider service, backend allocation, synchronous upload including
fence wait, and victim destruction. Provider service includes the host cache;
upload time is an API wall-time measurement, not a pure PCIe bandwidth timer.
These timers include failed calls, remain zero with profiling off, and do not
count service skipped by a device-cache hit. `clear()` also contributes eviction
time outside a projection request. Unpartitioned admission/bookkeeping and host
lease teardown remain in the enclosing request timer. The original `page_requests` and
`expert_requests` count logical selected pages, including device hits. Actual
host calls equal device `cache.provider_requests` plus host PLE requests;
the normal Rust counters still describe only host cache activity. Per-evaluation
metrics use counter/time deltas and retain capacity/current/lifetime-peak gauges.
`projection_backend` is null for the native reference, whose placement is
controlled by its own backend parameters.

## Validation and remaining gates

The mock suite uses two 1 KiB local shards and a synchronous fake backend. It
checks byte identity, shard/layout separation, allocation padding, eviction and
reload, active consumers, cache/provider/backend destruction, injected failures
and admission ordering. The ggml fixture uses explicitly initialized CPU kernels
to prove storage-to-backend upload, numerical computation and readback through
real backend allocations. Neither fixture calls the model loader or discovers
GPU devices. Exact results are recorded in device-paging-results.json.

On October 3 the targeted backend CTests passed **4/4**: device cache, ggml
adapter, existing host page provider and protected retention. The standalone
device-cache test also passed with llama.cpp disabled. The ggml fixture ran nine
matmuls with 18 scalar comparisons across F32, Q8_0 and IQ4_NL; maximum absolute
error was `1.90735e-05`, within the fixed `1e-5 + 1e-5 * abs(reference)`
per-value tolerance. Quantized payload readback was byte-exact. An initial
fixture incorrectly assumed Q8 dot-input scaling represented +/-1 exactly;
it now uses 0/+/-127, making that scale exactly one. The tolerance was unchanged.
These are tiny kernel fixtures, not model parity or GPU measurements.

The later graph-integration results are recorded separately in
device-graph-results.json, preserving the earlier
foundation evidence. Fresh backend and standalone builds passed **23/23** and
**14/14** CTests. Seventeen traced device-CPU cases and one untraced case passed;
18 invalid configurations were rejected. Across the traced cases, 1,372,852
floats and 10,394 IDs agreed exactly with each of the native and direct-CPU
references. The original graph suite also remained exact across 4,105,770 floats
and 33,286 IDs. These comparisons use generated, untrained fixtures only.

The F32/chunk-1 fixture illustrates actual cache behavior: 480 logical expert
requests produced 432 device hits and 48 host requests with the larger device
cache, versus zero hits and 480 host requests under forced eviction. Its larger
cache held only 393,216 B at peak. This verifies bypass and accounting, not a
checkpoint hit-rate or speedup prediction. The standalone projection fixture
passed 25 projections/50 scalar comparisons with maximum absolute error
`3.8147e-06` under its unchanged tolerance.

The first manual Vulkan projection fixture subsequently passed on an AMD Radeon
RX 6600 XT using RADV NAVI23: F32, F16, Q8_0, BF16, IQ4_NL, Q6_K, IQ4_XS and
IQ3_S, totaling 32 projections and 512 scalar comparisons. Maximum absolute
error was `7.6293945e-5`, within the fixed `1e-4 + 1e-3 * abs(reference)`
tolerance and `5e-4` NMSE bound. The independently sampled whole-device VRAM
peak was **1,068,335,104 B**, below the 4 GiB guard. This includes desktop and
other device allocations; it is not PMI's private allocation footprint.

Those initial tiny synchronous projections were slower on GPU: warm projection
times were approximately 0.23–0.38 ms versus 0.015–0.026 ms on CPU. The final
extended fixture covers both widths 256 and 2560: **16 cases, 64 projections,
1,024 values**, with unchanged tolerances. Maximum absolute error was
`0.000495910645` (0.30855 of its permitted scalar error). Self-reported peak RSS
was 75,583,488 B and sampled whole-device VRAM peaked at 1,073,295,360 B.
These results establish the upload/kernel/readback boundary, not real-model speed.

The full generated Qwen graph then passed **8/8** F32 and mixed Q8-expert/IQ4_NL-PLE cases,
with prefill chunks 1/4 and eviction/warm caches: 702,320 floats and 4,904 IDs per
CPU comparison, with exact tokens and routing IDs. This result uses a separately
labeled GPU criterion. The original Q8 CPU float comparisons passed **0/4**
because that CPU path requantizes activations. The uniform scalar criterion
passed **6/8** total cases: all four original-F32 cases and two of four cases
using an oracle that expands the exact stored Q8 weights to F32. The other two
oracle cases failed at a scalar affected by an FP16 cache midpoint. Only
`indexer_pool_k_new-*` now uses per-vector **NMSE <= 1e-7** (observed maximum
`2.805e-9`); all other floats retain `2e-5 + 1e-4 * abs(reference)`. Earlier
strict failures remain failures. The full GPU run peaked at 1,062,330,368 B
sampled device VRAM and 96,063,488 B group RSS, with no sampled swap growth.
See Vulkan evidence and numerical diagnoses for both
criteria and their limits; no trained-model correctness or speed claim follows.

The GPU fixture is deliberately not an automatic CTest. Its manual execution
requires `--profile-paging on` where applicable, the host guards in
[`tools/run_bounded.py`](../tools/run_bounded.py), and the nested whole-device
guard in [`tools/run_vulkan_smoke.py`](../tools/run_vulkan_smoke.py). The reviewed
test envelope is RSS/address space at most 12 GiB, available RAM at least 4 GiB,
owned swap at most 64 MiB, system swap growth at most 512 MiB, whole-device VRAM
at most 4 GiB, and a 600-second slot including startup and cleanup. These are
sampled limits, not hard driver-allocation bounds; profiling/allocation counters
do not replace them. See the linked GPU procedure for identity checks and logs.

Device expert retention and bounds on hidden scratch/staging remain outstanding.
No trained-model GPU test or performance claim follows from the tiny fixture.
The previous campaign remains stopped; the
next-day plan defines the remaining approval gates.

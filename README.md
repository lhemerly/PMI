# Poor Man's Inference (PMI)

Historical report names and unl inked investigation summaries below refer to private
receipts excluded from this public source proposal. Current evidence scope is
[ native cache invariance ](docs/native-cache-invariance.md).

Current native computation and evidence scope: [native cache invariance](docs/native-cache-invariance.md).

PMI experiments with explicit model-weight paging from storage into bounded RAM,
while retaining llama.cpp/ggml kernels. A small Rust core owns file access, cache
allocation and page lifetimes. C++ connectors own model layouts and computation.

`pmi run` now generates tokens through the **complete Qwen4exp CPU decoder**,
deferring routed expert matrices and PLE n-gram embedding rows to PMI storage.
The retained llama.cpp graph supplies attention, hyper-connections, routing,
shared experts and recurrent/KV state. `pmi reference` runs the ordinary backend
path for comparison.

Full-graph correctness has been demonstrated on complete, untrained synthetic
fixtures. The hash-verified Qwen3.8-Flash-Next checkpoint also passed a real-layer
comparison and two-token native demand-mmap reference comparison: 1,309,632
captured floats and 960 routing IDs agreed exactly. A separate bounded 16-token
CPU run completed at 0.914 decode evaluations/s with 5.757 GiB peak RSS.
These are short-run results, not long-context or comparative performance claims;
see trained-model validation and limits.
The release and optimization audit
documents experimental v0.1 gates and saved-trace cache-policy opportunities.
The Strata research note adds source-grounded ideas
for CPU/GPU overlap, bounded residency and storage batching, prioritized in the
roadmap
after the bridge. Compatibility and benchmark-comparison limits are explicit.
Optional protected expert retention and phase profiling
are implemented and fixture-validated. The routing dataset
stores bounded observations for later policy analysis. A reusable synchronous
[device cache and ggml adapter](docs/device-paging.md) provide the next backend
boundary. The Qwen decoder can select `device-cpu` or explicit `device-vulkan`
routed expert projections; direct CPU execution remains the default. Guarded
Vulkan checks now cover 16 projection cases and eight full tiny-Qwen
cases under a separately labeled numerical criterion; original stricter failures
remain recorded. A trained two-token Vulkan run completed within its resource
guards but failed native-CPU intermediate and routing parity. A bounded replay
reproduces its layer-0 failure exactly; the experimental CPU-compatible
activation mode passes that layer's frozen comparison. The preserved full-checkpoint CPU parity diagnostic for this mode failed;
CPU token parity is not promised by its floating Vulkan reduction. See the activation parity investigation
and Vulkan projection validation and limits.
Current work prioritizes GPU loading, residency and paging; CPU remains a
diagnostic reference; native resident Vulkan projections supply the paging oracle. The GPU-first runtime checkpoint
records completed dense loading followed by a host-pressure guard stop; no
completed replay or new performance claim resulted.
Durable progress telemetry now preserves
phase and cache-service evidence on an interrupted run; light fixtures passed. Earlier
admission evidence remains preserved.
Current generation is text-only, single-sequence, greedy and MTP-disabled.
Dense weights, PLE lookups and decoder state stay on CPU even with Vulkan
projections, so the host page-cache limit is not a bound on total RAM.

Implemented:

- GGUF inspection and validated multi-shard manifests, without reading payloads
- expert slices retaining shard, offset, length, shape and quantization
- a dependency-free Rust cache with aligned, zero-copy leases and counted pins
- byte limits that include alignment padding and outstanding page leases
- separate backend allocation budgets, synchronous uploads and device leases
- Qwen softmax/top-k routing, normalized weights, routed SwiGLU and gated shared expert
- existing ggml CPU kernels and optional Vulkan routed expert projections, including mixed quantization
- deferred expert/PLE allocation with graph hooks for selected matrices and rows
- chunked prefill, successive decode steps, and full-graph differential traces
- file-read/cache counters, process RSS/storage reads, and generation/layer timing

## Build and test

Install/use CMake 3.20+, a C++17 compiler, Python 3, and Rust/Cargo (tested with
Rust 1.95). The Rust crate uses only the standard library; Cargo builds offline.

Standalone storage/inspection tests do not need llama.cpp:

```sh
make test
```

For the paged CPU decoder, layer connector and reference decoder:

```sh
git submodule update --init vendor/llama.cpp
cmake -S . -B build -DPMI_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

llama.cpp is pinned at `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`, including recent
Qwen attention/pooling fixes. During configuration, CMake automatically applies
[`patches/llama-pmi-external-tensors.patch`](patches/llama-pmi-external-tensors.patch)
to that exact pin using `tools/apply_llama_patch.py`. Applying an already-present
patch is harmless; a different pin or conflicting vendor edits stop configuration.
The patch adds the deferred-storage/graph boundary; it is a local extension,
not an upstream paging API. An explicit Vulkan projection build requires the
Vulkan development tools and statically registered ggml backends. First create
the compiler wrapper described in the bounded Vulkan build procedure
at `build-vulkan/bin/glslc-serial`, then configure:

```sh
cmake -S . -B build-vulkan -DPMI_WITH_LLAMA=ON -DGGML_VULKAN=ON \
  -DGGML_BACKEND_DL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_GLSLC_EXECUTABLE="$PWD/build-vulkan/bin/glslc-serial"
```

Run `cmake --build build-vulkan -j1` inside the documented host supervisor.
Outer `-j1` alone does not limit the shader generator's internal compiler workers.

CPU selection remains explicit and does not initialize Vulkan. GPU tests are
manual, guarded runs; the `pmi_vulkan_projection_test` executable is deliberately
not registered with CTest. The ordinary CTest suite keeps its CPU/mock workloads.
See the Vulkan test guards before executing a GPU
fixture. Backend support in `reference` is separate from PMI projection placement.

## Inspect a model

Supply every shard, in any order:

```sh
./build/pmi inspect model-00001-of-00003.gguf model-00002-of-00003.gguf model-00003-of-00003.gguf > manifest.json
```

The JSON includes canonical shard paths and tensor offsets local to each shard.
Missing shards, conflicting metadata, duplicate tensors and invalid ranges fail
before inference. Name-based semantic classes are advisory; the connector
validates exact tensor names and layouts before execution.

## Generate through PMI paging

Create a complete tiny fixture without downloads or external Python packages:

```sh
python3 tools/make_qwen_fixture.py build/fixtures/qwen/tiny.gguf --shards 2
./build/pmi run \
  build/fixtures/qwen/tiny-00002-of-00002.gguf \
  build/fixtures/qwen/tiny-00001-of-00002.gguf \
  --prompt 'hello world' --max-tokens 8 --prefill-chunk 4 --context 64 \
  --cache-bytes 8192 --trace-json build/fixtures/qwen/paged-trace.json
```

This fixture exercises three gated-delta-net layers, one QSA attention layer,
hyper-connections, four routed experts with top-two selection, shared experts,
and PLE history. Its untrained weights produce `8y3t.vl7` for this prompt; that
output is a reproducible test result, not a language-quality example. The
8,192-byte cache forces expert eviction and reload. Supply every shard to `run`;
paths may be renamed and supplied in any order.

Compare the same model with the ordinary reference path:

```sh
./build/pmi reference build/fixtures/qwen/tiny-00001-of-00002.gguf \
  --prompt 'hello world' --max-tokens 8 --prefill-chunk 4 --context 64 \
  --gpu-layers 0 --trace-json build/fixtures/qwen/reference-trace.json
```

`reference` accepts only the first shard and uses the backend's conventional
shard-name discovery. It does not use PMI page storage.

Both commands print generated text, a `TOKENS` line and JSON metrics prefixed
by `PMI_METRICS` or `REFERENCE_METRICS`.

The fixed-token cache comparison
records native Vulkan timing partitions and the 256 MiB/1.5 GiB cache experiment.
The larger cache has one completed sample; its resumed repeat stopped on the
unchanged host memory-pressure guard before prefill. Repeat variability remains
unresolved, and the measured hybrid path is slower than the CPU control.

| Option | Behavior |
| --- | --- |
| `--prompt TEXT` | Required prompt; tokenization uses the GGUF vocabulary. |
| `--max-tokens N` | Maximum generated tokens; default 32. Generation stops at an end token. |
| `--teacher-forcing-file PATH` | Optional benchmark replay: a regular file <=65,536 bytes containing exactly `max-tokens` whitespace-separated nonnegative vocabulary IDs. Feeds and emits these IDs, including end tokens; prints independent greedy `PREDICTIONS` and match count. Empty/invalid files fail. Supplied text is not a generation-quality result. |
| `--prefill-chunk N` | Prompt tokens per decode call; default 1. |
| `--context N` | Requested context, defaulting to prompt tokens plus generation limit. Must fit that total and not exceed the model's training context. |
| `--cache-bytes N` | `run` only; default 67,108,864. Must fit the largest requested expert matrix or PLE row after 64-byte rounding. |
| `--cache-policy lru\|previous-top4` | `run` only; default `lru`. Protect the previous evaluation's final-token top-four expert bundles per layer, reducing the quota when capacity requires it. |
| `--profile-paging on\|off` | `run` only; default `off`. Time paged requests, copies, graph setup/teardown, kernels and retention separately. |
| `--dense-load default\|direct-required` | `run` only; default unchanged. Linux `direct-required` aborts if resident dense payloads cannot be read with direct I/O. Per-shard counters distinguish logical payload, aligned reads and staging; metadata and ordinary PMI page reads remain buffered. |
| `--dense-pages native\|base` | `run` only; default `native`. Linux `base` advises complete pages inside owned dense CPU buffers against THP. Requires direct-required loading, zero GPU layers and CPU repacking off. |
| `--load-trace-jsonl NEW_PATH` | Optional exclusive dense-loader allocation/tensor/read/copy trace, bounded to 16 MiB/65,536 records. Requires the same loader configuration as base advice; adds timing and I/O overhead. |
| `--device-upload-host-policy retained/transient` | Opt-in upload-only host leases; retained is the default. Live transient allocations remain inside the host quota. |
| `--device-upload-host-io buffered\|direct-preferred\|direct-required` | Device projection only; default `buffered`. New upload source misses can use aligned direct reads. Preferred permits an explicit unsupported-path fallback; required rejects it. Payload plus temporary aligned staging must fit the host quota. Cached host/device hits keep their existing behavior. Kernel filesystem buffering is not ruled out by required mode. |
| `--progress-jsonl NEW_PATH` | `run` only; exclusive, bounded process-interruption progress. Persists loading, evaluation completion and device-cache read/allocation/upload stages; no model text or activations. Detail truncation is explicit. |
| `--projection-backend cpu\|device-cpu\|device-vulkan` | `run` only; default `cpu`. Device modes upload compact selected expert matrices and compute synchronously using the explicitly chosen backend. Both require LRU and the two device budgets below. |
| `--projection-dispatch all\|split` | `run` only; default `all`. `split` seeds free device-cache capacity during the first evaluation, then freezes it: one grouped worker computes resident hits while existing CPU kernels compute misses, with an explicit join before output scatter. Requires a device backend, LRU, and native activation mode. |
| `--projection-activation backend-native\|cpu-compatible` | `run` only; default `backend-native`. Experimental `cpu-compatible` requires `device-vulkan` and IQ3_S/IQ4_XS/IQ4_NL/Q8_0 expert weights; prepares native CPU quantized activations while keeping matrix multiplication on Vulkan. Validated on the captured layer-0 replay, not the full checkpoint. |
| `--projection-device-index N` | `device-vulkan` only; Vulkan device index, default 0. Selection errors fail without CPU fallback. The selected identity must match the independently guarded physical device. |
| `--device-cache-bytes N` | Required for either device mode; positive aligned backend weight-cache budget, independent of `--cache-bytes`. CPU-backed allocations consume extra host RAM; Vulkan counts managed backend buffers. |
| `--device-graph-bytes N` | Required for either device mode; positive budget for one projection's aligned input/output backend buffer. Excludes hidden backend scratch, metadata and host vectors. |
| `--routing-jsonl NEW_PATH` | Optional Qwen routing IDs/weights and metrics, with no token text/IDs or hidden activations. Exclusive file, 8 MiB/65,536-record bounds; incomplete/truncated captures are explicit. |
| `--gpu-layers N` | Default 0. `run` requires 0, including with Vulkan projections; outer/dense-layer offload is separate. `reference` accepts backend GPU placement, including -1. |
| `--trace-json PATH` | Optional tokens, per-evaluation graph snapshots and metrics. Trace payload is capped at 64 MiB; tracing adds memory and timing overhead. |
| `--trace-mode full\|compact` | Default `full`; `compact` retains routing, layer outputs, PLE boundaries, FFN inputs and final logits without large expert intermediates. |
| `--cpu-repack on\|off` | Default `on`; use `off` in both comparison paths to keep their dense kernels matched. |
| `--reference-mode default\|demand` | `reference` only. `demand` uses native CPU graph operations with ordinary mmap demand faults, disables initial whole-shard prefetch and CPU repacking, and uses no PMI callbacks. It does not enforce a memory limit. |

The direct-I/O and pressure-guard preparation
documents mandatory dense loading, optional zram/swap-out/PSI stops, and the
admission budget for a separately approved short checkpoint validation.
The direct expert-miss correctness results
describe the opt-in provider path, quota/lease/error checks, exact tiny-model
comparisons and the remaining Btrfs cache-bypass limitation.
The owned-buffer base-page experiment
records fixture parity and a completed 16-prediction trained CPU run. Hybrid
priming later stopped on the unchanged host swap-out guard; paired CPU/hybrid/
split timing remains incomplete. Optional loader diagnostics add measurable overhead.

The generator also accepts `--expert-type f16` or `q8_0`, and `--ple-type f16`,
`q8_0` or `iq4_nl`. Quantized PLE variants use wider fixture embeddings to satisfy
the format's row-block size. All generated weights are synthetic.

The opt-in bridge keeps PLE lookups, dense weights and the outer decoder on CPU.
It uploads only the selected compact expert matrix, and device-cache hits skip
host page requests. `device_projection` metrics separate its uploads, hits,
actual provider calls and graph-buffer charges from the existing host counters.
It rejects `previous-top4` because that policy protects only host allocations.
Only `device-vulkan` selects a GPU. Its `vram_bytes` is null because these
generation metrics do not sample physical VRAM; managed cache/graph charges
exclude driver heaps, staging and scratch. Manual GPU tests here require phase
profiling, host-memory/swap/time guards and a separate whole-device VRAM guard.
See [the device graph boundary](docs/device-paging.md) and
Vulkan evidence and guard procedure.

The full-model test runs automatically through CTest, or directly:

```sh
python3 tests/full_model_test.py ./build/pmi build/fixtures/full-model
```

Fixed-token replay has a separate fixture test. It changes a decoder input,
checks the resulting logits against resident execution, and checks invalid IDs:

```sh
python3 tests/teacher_forcing_test.py ./build/pmi build/fixtures/teacher-forcing
```

Add `--vulkan` only when supervising the test with the documented host and
whole-device resource guards. The replay fixes decoder token inputs; different
backend arithmetic can still change routing and cache requests.

It checks 22 original differential cases, eight reference/tracing mode cases,
eleven retention/profiling cases and four ordinary runs without tracing.
Coverage includes
F32/F16/Q8_0 experts, IQ4_NL PLE rows, forced eviction, renamed/reversed shards,
EOS history resets and longer QSA prompts with prefill chunks of 1, 4 and 64.
Retention tests preserve routing, greedy tokens and intermediate values, exercise
quota fallback and pin cleanup, and compare live routing captures with full traces.
Results are saved in `build/fixtures/full-model/full-model-results.json`.

## Execute a paged layer fixture

The test suite preserves synthetic fixtures under the build directory. This
runs a rectangular layer using the target's mixed quantization types:

```sh
./build/pmi moe-layer \
  build/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/qwen-moe-00001-of-00002.gguf \
  build/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/qwen-moe-00002-of-00002.gguf \
  --input build/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/input.txt \
  --cache-bytes 18432 --repeat 2
```

Use `--cache-bytes 4194304` to retain all requested fixture pages and observe
warm hits. Each output line is JSON containing routing IDs/weights, output,
file-read bytes, cache usage, latency and process memory. `--layer N` selects a
layer; top-k and expert scaling come from canonical GGUF metadata. An explicit
`--experts-used N` can override top-k for controlled experiments.

The input is one whitespace-separated floating-point embedding. This command
has no tokenizer, attention, recurrent history or PLE state. Separate gate/up/down
layouts are supported; fused expert gate/up, external scales, alternate routing,
and batched inputs are rejected. Source shards must remain immutable while open.

`io_bytes` measures bytes returned by file reads, including OS-cache hits.
`process_storage_read_bytes` is the Linux process storage counter, or null when
unavailable. The first iteration has an empty PMI cache; it is not an OS-cold
storage benchmark. Cache limits exclude graph/state allocations and the OS cache.

## Measurement scope and real-model gate

Generation metrics separate setup, prefill and decode time, and include expert
and PLE requests, deferred weight bytes, cache hits/misses/evictions and process
memory. They also report time to the first sampled token (including setup),
time to that token after setup, prompt tokens, and per-evaluation logical file
bytes/cache counters, physical storage-read deltas and RSS. This timestamp is
internal generation latency; the CLI prints text after the run completes.
The first generated token comes from the final prefill evaluation, so
decode evaluations need not equal the number of generated tokens. Tracing and
synthetic model sizes make these correctness runs unsuitable as real-model
throughput or SSD-performance estimates.

See [architecture](docs/architecture.md), [validation evidence](docs/validation.md),
and the weight/resource proposal. Target identities,
licenses, exact revisions and hashes are in target evidence.
No target weights are included in Git. The approved 93,682,584,224-byte checkpoint
completed full SHA-256 verification on 2026-10-02 at 22:02:35 UTC. The license
and model card are retained alongside the local weights. Four initial guarded
attempts and one explicitly approved 512 MiB system-growth trial stopped on
global swap growth. A separately approved instrumented trial then completed:
`Hello` produced `, I` (token IDs `11 353`) through PMI paging, at 5.90 GiB
reported peak RSS. See the completed trial and
earlier target attempts. The later
three-stage validation established native
two-token intermediate/routing/token parity and completed a 16-token run.
Long-context correctness, controlled sustained-performance comparisons and
multi-sequence execution remain outstanding. Later native Vulkan paging passed
the bounded cache invariance gate described in [native cache invariance](docs/native-cache-invariance.md);
broad checkpoint/quantization compatibility and model quality remain unqualified.
The trained Vulkan revalidation
completed within its resource guards, but the opt-in `cpu-compatible` activation
mode still failed the original intermediate/routing parity gate. Its two greedy
tokens matched; that result does not qualify the GPU path as correct. The
conditional GPU/CPU throughput comparison was therefore not run.
After the user clarified that exact CPU parity is a diagnostic rather than a
unique arithmetic oracle, a provisional GPU/CPU baseline
completed: 0.635 versus 1.237 decode tokens/s over 15 decode evaluations each.
The sampled layer's native Vulkan arithmetic is closer to a Float64 reference
than CPU activation emulation. The runs have different generated continuations
and OS-cache histories, so these are provisional measurements, not a controlled
speedup or model-quality result.
The later cache campaign remains stopped;
the tiny GPU tests do not resume it. These results do not establish model
quality or SOTA performance.

`tools/download_qwen_weights.py` plans the pinned inventory without networking by
default. Explicit `--download` enables resumable transfers, checksum verification
and local license/model-card retention. Do not run two downloaders against the
same partial file; `--shard 1`, `2`, or `3` limits the selected shard.

`tools/run_bounded.py --help` describes the Linux command supervisor used for
large-model validation. Its RSS/available-memory/swap checks are sampled and do
not prevent every allocation burst. Optional address-space limits are hard per
process and must account for mmap's virtual size. Keep model runs sequential;
the ordinary unguarded reference loader is inappropriate for this oversized
checkpoint because it can prefetch and repack much more than physical RAM.
`--max-system-swap-growth-bytes` can explicitly separate host-wide swap growth
from the owned-process `--max-swap-growth-bytes` cap. If omitted, both caps retain
the original value. The first explicitly approved 512 MiB system-wide trial
stopped during loading. The later instrumented trial used the same limits and
completed two tokens, with sampled global growth 507.57 MiB and zero owned swap.
Both single-trial approvals are consumed. The subsequent three-stage approval
is also consumed; it retained the same physical-memory and swap limits and
explicitly allowed 128 GiB virtual address space for the native mmap reference.

The runner also logs observational zram allocation/compression, swap activity,
memory PSI and shared-cgroup accounting. These do not change its stop conditions.
See memory telemetry and trial policy for units,
missing/reset semantics, idle validation and the policy used for the completed trial.

Transient upload leases and baseline reference comparison record the opt-in policy, light correctness fixtures and real-model parity limit.

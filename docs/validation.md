# Local implementation validation

Historical report names and unl inked investigation summaries below refer to private
receipts excluded from this public source proposal. Current evidence scope is
[ native cache invariance ](native-cache-invariance.md).

Current native computation and evidence scope: [native cache invariance](native-cache-invariance.md).

Validated 2026-10-02 in the isolated local branch `codex/pmi-runtime-core-qwen`,
based on PMI `75f798f`. No other project was edited; no push, merge or deployment
was performed. The fetched checkout was initially clean and contained no local
PMI edits or project-level `.agents/skills`. Vendor instructions were inspected.
The first storage/layer checkpoint is preserved in local commit `ccca3ef`.
The full-graph continuation adds the tracked CPU loader/graph patch described in
`dependencies.md`; it does not publish or change the upstream submodule pin.

## Results

The separately approved 2026-10-03 trained-model sequence passed all three
bounded stages: real layer-0 comparison, exact two-token native demand-mmap
parity, and 16-token untraced CPU generation. See the
current results and scope and
machine-readable evidence.
The earlier fixture and stopped-run results below remain historical evidence.

The full-graph CPU checkpoint at llama.cpp
`4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe` plus the tracked PMI patch passed all
**14 then-registered CTest tests**:

1. Fixture generation.
2. Legacy GGUF/expert-slice self-test through the new provider.
3. Bounded remote-header reader, including range/budget failures (offline test).
4. Rust runtime: **12 unit tests**.
5. Multi-shard manifest tests, including **43 rejection cases**.
6. C++/Rust PageProvider bridge lifetime, alignment, shard, pin and I/O cases.
7. Serialized quantization sizes against the pinned ggml definitions.
8. Qwen MoE connector against independent resident ggml graphs and F32 scalar math.
9. Public CLI golden outputs, reordered shards, warm reuse and missing-shard errors.
10. Paged graph operators: seven storage types, strides/broadcasts, graph reuse,
    bounded traces, allocation planning, latched failures and consumer abort.
11. Complete synthetic Qwen decoder: 22 differential cases and four runs without
    tracing, including quantized PLE, chunked prefill, decode, history and eviction.
12. Pinned downloader: seven offline metadata, hash, resume, publication and
    refusal tests, with network calls blocked.
13. Bounded command supervisor: ten small-process/predicate memory, timeout, exit and
    owned-group cleanup tests; unrelated processes remain untouched.
14. Independent complete packed-MoE reference against a mixed-quantization fixture.

The backend-free `make test` build also passes all 8 applicable CTest tests.
Rust formatting and strict Clippy passed. GGUF/manifest ASan+UBSan checks passed;
LeakSanitizer could not run under the environment's ptrace, so no leak-sanitizer
result is claimed. Ordinary pointer/lifetime/lease tests did run.
After the trained-model guard stops, the supervisor's separate system-swap cap
was verified with its ten tests (1.975 s); C++ and Rust inference code did not change.
The later observational telemetry patch passes 13 supervisor tests and 20 new
offline telemetry tests (two CTest groups, 2.98 s). A seven-sample, 30-second
read-only observation validates available host/zram/cgroup signals; see
telemetry evidence and limits. This patch changes no
inference code, guard thresholds or machine settings and adds no trained-model run.

Commands used:

```sh
make test
cmake -S . -B build-baseline -DPMI_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure
cargo clippy --manifest-path runtime/Cargo.toml --offline --all-targets -- -D warnings
cargo fmt --manifest-path runtime/Cargo.toml --check
python3 tools/inspect_remote_headers.py --self-test
```

The registered tests regenerate runnable shards/input/golden outputs under
`build-baseline/fixtures/moe/` and measurements in
`build-baseline/fixtures/cli-metrics.json`. Compact captured results for the original layer checkpoint and full decoder
are retained privately. The latter is derived from the final CTest output at
`build-baseline/fixtures/full-model/full-model-results.json`.

The backend patch also applies cleanly to a fresh archive of the exact pin. All
13 modified tracked files and the added observer header match the compiled checkout byte for byte, and the reverse
patch check passes. The patch script performs no network access or reset.

## Full decoder comparison

`pmi run` now executes the complete Qwen4exp graph with deferred expert and PLE
storage. The generated, untrained four-layer fixture includes HC, three recurrent
GDN layers, one sparse-attention layer with indexer pooling, shared/routed experts,
PLE ngram history and tokenizer/output tensors. Its F32 file is 800,416 bytes,
with 113 tensors; it is not a downloaded or trained checkpoint.

Four complete variants cover F32/F32, F16/F16, Q8_0/F32 and Q8_0/IQ4_NL expert/PLE
storage. The resident and paged paths compare greedy token IDs and integer
routing/indexer tensors exactly. Floating-point MoE, PLE, indexer and final-logit
traces use `abs(error) <= 2e-5 + 1e-4 * abs(reference)`. Across 22 cases,
**2,053,178 floats and 16,634 integers matched with zero observed difference**.
Four additional runs disable tracing to exercise normal scheduler partitioning.
Eight option cases additionally verify compact/full trace agreement, exact FFN
inputs, demand-reference behavior, repacking controls and metric reconciliation.

Coverage includes prefill chunks 1/4/64, retained-cache and forced-eviction runs,
single/split resident references, arbitrarily renamed and reversed shard paths,
EOS history resets, and a longer prompt crossing the sparse-attention pool budget.
Missing shards and an insufficient cache fail without a successful token report.
The direct operator suite separately compares 230,400 values across F32, F16,
Q8_0, IQ4_XS, IQ3_S, IQ4_NL and Q4_0, including the real target's 90-byte PLE row
layout, with zero observed error. It also proves that a latched partial-progress
read failure aborts before a downstream consumer modifies its sentinel output.

For prompt `hello world`, the F32 fixture generates synthetic text `8y3t.vl7`
and IDs `298 284 293 279 286 281 271 297` in both paths. Those tokens demonstrate
stateful graph execution, not language quality.

Selected measurements from the final full-model run (CPU, OS-cache warm):

| Measurement | Retained cache | Forced eviction |
| --- | ---: | ---: |
| Prompt prefill chunk | 1 | 1 |
| Generated tokens | 8 | 8 |
| Deferred tensor payload | 397,568 B | 397,568 B |
| Cache capacity / peak | 4,194,304 / 395,008 B | 8,192 / 8,192 B |
| Logical file bytes read | 395,008 B | 3,934,720 B |
| Hits / misses / evictions | 444 / 76 / 0 | 0 / 520 / 519 |
| Whole-run time including loading | 23.828 ms | 25.362 ms |
| Prefill / decode time | 9.392 / 4.179 ms | 10.263 / 5.000 ms |
| Peak RSS | 18,010,112 B | 19,578,880 B |
| Process storage reads / GPU allocation | 0 / 0 B | 0 / 0 B |

The longer forced-eviction case reads 11,213,952 logical bytes and evicts 1,481
pages while preserving the same 8,192-byte allocation bound. Normal F32 execution
without tracing (chunk 4) takes 17.183 ms and peaks at 17,018,880-byte RSS in this
sample. These tiny-model timings include initialization and optional tracing;
they are diagnostics, not predictions for the 93.68 GB target. The JSON records
decode evaluations per second, distinguishing seven decode evaluations from
eight generated tokens (the first token comes from prefill).

## Numerical comparison

Fixtures cover F32, Q8_0, IQ4_XS, IQ3_S, IQ4_NL, and mixed IQ3_S gate/up + IQ4_NL
down + Q8_0 shared weights. The mixed fixture uses rectangular shapes
(embedding 256, routed hidden 128, shared hidden 64), two shards with overlapping
local offsets, and two CPU threads in an additional eviction pass.

Tests compare selected expert IDs exactly and compare router logits,
probabilities, weights, gate/up/SwiGLU/down intermediates, shared gating and final
layer output using `abs(error) <= 3e-7 + 3e-5 * abs(reference)`.
Quantized fixtures, including changed routing and eviction/reload, matched the
resident ggml results exactly on this CPU. The largest F32/scalar difference was
`1.19209e-7`, within 5.36% of the allowed per-value tolerance. These results do not
establish quantization quality relative to the unquantized real model.

Changing the input changes selected IDs from `[1,2]` to `[3,0]`; both cases are
compared independently. Missing slices, unsupported layouts/scales, capacity
failure, corrupt files and non-finite inputs are rejected without leaked leases.

## Measured layer telemetry

For the rectangular mixed fixture:

| Measurement | Result |
| --- | ---: |
| File bytes read on the first pass | 150,528 |
| Additional file bytes on retained-cache repeat | 0 |
| Warm hits / misses | 11 / 0 |
| Forced-eviction cache capacity / peak allocation | 18,432 / 18,432 bytes |
| File bytes on each forced-reload pass | 150,528 |
| Retained-cache cold / warm latency sample | 1.882810 / 0.168395 ms |
| Cold / warm layer evaluations per second | about 531 / 5,938 |
| Warm process RSS | 11,108,352 bytes |
| GPU allocation | 0 bytes; CPU path |
| Linux process storage-read counter | 0 bytes for these fixture runs |

The fixture files were already in the OS cache. File-read counters therefore
show useful payload traffic, not physical SSD traffic. The first pass includes
CPU kernel initialization; these single samples are diagnostic, not controlled
benchmarks. Layer evaluations per second are not model tokens per second. Cache
allocation numbers include 64-byte padding but exclude graph/state, allocator
metadata and OS-cache memory; RSS is reported separately.

## Real-model reference smoke test

The already installed `Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf` was rerun
after full-graph integration and generated the same eight baseline tokens with
the normal llama.cpp loader and the same whole-prompt prefill (five tokens):

```sh
./build-baseline/pmi reference \
  /path/to/models/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf \
  --prompt 'The capital of France is' --max-tokens 8 --gpu-layers 0 --prefill-chunk 5
```

Output: ` Paris. Which of the following statements is`

IDs: `12095 13 15920 315 279 2701 12239 374`

Whole-run time including loading: 2773.227 ms. Peak process RSS: 7,877,505,024
bytes. Process storage reads: 1,527,808 bytes (largely OS cached). The first
checkpoint's corresponding time was 5802.34 ms with 4,689,051,648 storage-read
bytes; different cache state prevents a speedup comparison.

With prefill chunk 1, this quantized resident model instead ends with `options
correctly` (IDs `2606 12440`) after the same first six tokens. Prefill chunking can
change numerical execution and greedy choices; comparisons must keep it matched.
Full synthetic reference/paged comparisons do so. This local model tests the
retained reference path, not Qwen3.8 validation or PMI paged decoding. Its file
was neither changed nor downloaded.

## Download and guarded trained-model attempts

All three approved UD-IQ4_XS shards, totaling 93,682,584,224 bytes, passed full
SHA-256 verification at **2026-10-02 22:02:35 UTC**. The exact revision is
`38bb39ee97821de2c9009abb7e93950eec396e66`. The official license and model card
are retained locally. An executor disconnect interrupted the original transfers;
after confirming they had stopped, the same partial files were resumed. No second
full copy or alternative quantization was downloaded.

The first PMI attempt used CPU only, MTP disabled, `Hello`, two requested tokens,
context 256, prefill chunk 1, a 512 MiB cache, repacking off and compact traces.
The supervisor enforced 12 GiB RSS, at least 4 GiB available host memory, 64 MiB
owned/system swap and ten minutes; PMI had a 12 GiB address-space cap. Core dumps
were disabled only for each child process. Global swap growth stopped loading.
After a 30-second idle observation with no additional swap-outs, one retry used
the identical limits and also stopped during loading. No further PMI retry ran.

The native demand-reference attempt used the same prompt, requested token count,
context, chunk, repacking and trace settings. It used native ggml operations,
no PMI callbacks, and a 128 GiB virtual-address allowance for the file mappings.
It initialized context/graph buffers but also hit the global swap-growth guard.
The smaller independent layer check then stopped under a 2 GiB RSS/address cap.

| Attempt | Peak sampled RSS (GiB) | Minimum available RAM (GiB) | Peak global swap growth (MiB) | Guarded elapsed (s) |
| --- | ---: | ---: | ---: | ---: |
| PMI initial | 3.628 | 10.654 | 75.344 | 4.007 |
| PMI identical-limit retry | 3.773 | 10.985 | 199.449 | 4.008 |
| Native demand reference | 4.008 | 14.253 | 177.816 | 6.007 |
| One-layer reference | 0.250 | 14.541 | 224.309 | 2.004 |

Every attempt had zero sampled owned-process swap and exited after SIGTERM;
none needed SIGKILL or left a live owned process. There was no reported OOM or
allocation error. These are partial aborted-run observations, **not** full-model
RAM requirements, first-token latency or throughput. No completed token report,
trace comparison or layer comparison was produced. Exact commands, limits,
preflight values, I/O counters and final guard records are preserved in
target-validation-results.json; full local logs
are under `build-baseline/target-20261002/`.

The host's existing swappiness was observed as 150. Global growth may include
model-induced reclaim of other applications' pages or unrelated activity; zero
owned swap does not distinguish them. No host setting was changed. The supervisor
now supports an independent `--max-system-swap-growth-bytes`, with the original
default thresholds retained.

### Single approved 512 MiB system-growth trial

At 22:25:13 UTC the user explicitly approved one trial changing only the global
growth allowance to 512 MiB. It began at 22:26:31 UTC with the same PMI command,
64 MiB owned-swap cap, 12 GiB RSS/address caps, 4 GiB available-RAM floor and
600-second timeout. Independent review confirmed all effective limits matched.

At **2.003 seconds**, system-wide growth was **539,877,376 bytes (514.867 MiB)**
and the guard sent SIGTERM. The final peak during termination was 617,910,272
bytes (589.285 MiB); the command ended at 3.005 seconds with return code -15,
and the supervisor returned 125. There were no live owned processes, SIGKILLs,
monitor errors or reported allocation/OOM errors.

Peak sampled RSS was 976,097,280 bytes (0.909 GiB), minimum available RAM was
15,704,973,312 bytes (14.626 GiB), and owned swap stayed zero. The loader had
reported its 5103.71 MiB planned CPU buffer but had not finished loading it.
No context/decode/token result or trace was produced. Physical storage reads
were zero in this OS-cache-warm attempt; partial `rchar` was 874,382,250 bytes.
Neither value is complete inference I/O, and no throughput can be calculated.

This result establishes another global-limit crossing, not its cause or full
model memory fit. The single approved trial is consumed. **No retry or further
limit increase was performed.** No machine settings were changed.

Model runs and hashing were sequential. No system cache flush was performed.

## Completed instrumented trained-model trial

After the telemetry-only commit `704762d`, the user explicitly approved one
instrumented repeat at the same limits. On 2026-10-02 at 23:24:57 UTC, prompt
`Hello` generated `, I` (`11 353`) through PMI paging and exited successfully.
Reported peak RSS was 5.903 GiB, with 2,253 evictions and zero cache hits under
the 512 MiB cache. One decode evaluation took 1.298 seconds (0.77056 evaluations/s).

Sampled global swap growth reached 507.574 MiB, only 4.426 MiB below its cap.
Measured zram allocator growth was 135.902 MiB, owned swap stayed zero, and host
full-memory PSI increased by 1.004 seconds. No OOM or guard event occurred.
Finite/shape, routing/softmax, argmax and paging-accounting audits passed on the
saved trace; a trained-reference comparison was still outstanding at that time.
The later three-stage validation completed it for
the two-token case.
See the full result and limits. The single approval
is consumed; there was no retry, settings change or further limit increase.
An empty PMI cache is not evidence of physically cold SSD reads, and the retries
have different OS-cache histories.

`pmi_target_moe_reference` supplies a smaller independent check. It loads the
complete physical FFN tensors for one layer, applies native batched
`MUL_MAT_ID`, and compares them against PMI using the same input. Layer 0 needs
1,203,230,720 tensor bytes. The harness rejects plans exceeding its 2 GiB
weight/workspace/cache allowance before allocation. Twelve small fixture cases
passed (32,472 float values, max error 5.96e-8), and eight malformed/resource
cases were rejected. An actual FFN input is captured as `ffn_input-L` during
full decoding. The initial target-layer attempt instead used a deterministic
synthetic 2560-value vector because no complete decoder trace was available;
it was stopped before comparison. The later approved layer-0 run used both saved
real inputs and passed 110,872 native/connector float comparisons, plus the
saved-graph boundary comparison. Layer parity is conditional on its input;
the separate two-token native decoder comparison supplies full-graph coverage.

## Not yet established

The bounded demand reference has completed real-target two-token generation,
with exact captured logits, routing and token parity. Tokens 3–16 from the
longer PMI run remain reference-unverified. Neither this shared-backend parity
nor synthetic fixtures establish independent official-implementation correctness,
quantization quality or controlled comparative performance. A fully resident
target still exceeds physical RAM; demand-mmap completion does not establish
full residency.

Native Vulkan paging is implemented and has passed the bounded trained-checkpoint
cache invariance checks in [native cache invariance](native-cache-invariance.md).
Asynchronous I/O, concurrent sequences, long-context memory growth and comprehensive
independent backend/state allocation accounting remain unqualified.
Only PMI page allocations have an in-process enforced byte limit; the external
supervisor samples total RSS and can miss bursts. CPU expert projections use serial
one-matrix ggml kernels and are intended for correctness before optimization.
There is no SOTA or comparative performance claim. Weight identity, model license
and machine-specific admission receipts are retained privately; this source proposal
does not include weights or a portable resource-fit guarantee.

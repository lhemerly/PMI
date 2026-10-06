# Native cache invariance (2026-10-06)

The experimental paged runtime uses native Vulkan for all routed expert projections under dispatch=all; routing, shared experts, dense/attention and recurrent graph work remains on CPU. Backend-native retains the pinned backend's type/shape/precision contract. CPU token agreement is a diagnostic, not a uniquely correct arithmetic definition. Split dispatch is a separate policy because hits use Vulkan and misses use CPU.

A fixed nine-evaluation native A1→B→A2 check passed exact captured-record, complete-logit, weighted-routing, prediction and source-order invariance. A uses 2 GiB device cache; B uses 2 MiB forces capacity churn. Both references and B agree on 2,619 compact records, 2,234,880 vocabulary logits and 432 weighted routing records per run. B recorded 12,959 capacity evictions and 12,960 uploads; each reference recorded 3,194 evictions and 5,845 uploads. Same original tensor bytes/descriptors, native single-vector kernels, precision, input prefix/context and direct-required transient staging were retained. Local raw evidence is retained privately; no checkpoint payload or raw host telemetry is published here.

A separate 33.8 MB real-input layer replay checked full ten-expert IQ3_S gate/up and IQ4_NL down vectors against direct resident allocations on the pinned locally built backend. Cold/warm/explicit-clear reupload and buffered retained/transient staging passed 258 comparisons. Its local concatenate/combine helper was fixture-scoped; the full-model A-B-A above exercised the actual graph integration with direct-required I/O and capacity eviction.

The pinned llama.cpp base is 4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe with the committed external-tensor/precision-extension patch. The native oracle bypasses PMI paging but uses the same locally built backend; it is not an independently built pristine upstream reference. The experimental cpu-compatible option quantizes to CPU Q8_K/Q8_0, expands to F32 and uses Vulkan floating reduction. It does not promise CPU dot or prediction equality. In the preserved case CPU predicted 852 and native/compatible predicted 440 at index 8; native/CPU maximum logit difference 0.37810874 remains diagnostic. No tolerance was relaxed.

These tests cover one checkpoint, prompt/prefix and driver configuration. They establish scoped integration/cache invariance, not general quality, correctness for every quantization type or prompt, CPU parity, whole-model GPU execution, sustained throughput or production latency. Native 1.5/2 GiB capacity timing remains separately instrumented and requires exact output/routing gating before interpretation. The attempted native capacity ABBA timing sequence stopped after B2 failed its
postflight GPU-idle gate. A2 was not run; the incomplete sequence supports no
capacity effect percentage or speedup claim. Independent compact-evidence review
verified ordered streams and tensor hashes against previously verified receipts;
it did not independently replay raw floats from the compact artifact.

Model downloads and compute windows require explicit resource admission; guards, pin/patch provenance and reproducible byte-range checks remain necessary.

See [architecture](architecture.md), [dependencies and patch setup](dependencies.md) and [device paging](device-paging.md) for source/build details. Historical benchmark and host-telemetry reports referenced in older prose are private and are not included in this public source scope.

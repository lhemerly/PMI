# Inactive exact-text smoke and bounded workload proposal

The one-call smoke asks: `Return exactly this one lowercase word: baseline.` Its trusted evaluator is `exact-text-v1`, expected text `baseline`, maximum 64 UTF-8 bytes. It compares the entire returned string without trimming, normalization or case folding. Extra text fails. The exact prompt, argv, source and configuration identities are recorded in LIVE-SMOKE-PROPOSAL.json. This is an integration smoke, not evidence of research quality.

One-call command (not executed; separate legitimate approval and qualification required):

```sh
flock -n /tmp/cpu-coordinator-6t3yfB/test.lock timeout 260 nice -n 10 python3 -B content_factory.py research-start --spec-file research/live-smoke-spec.json --out .research-live-smoke-001 --lease /home/luizh/Documents/Codex/2026-10-08/task-8/content-factory-shared.lease
```

One released proposal is evaluated, then the run stops before review with `budget_exhausted`, exit 3. It does not complete the loop.

## Single bounded workload for review

BOUNDED-WORKLOAD-POLICY.json is an inactive proposed policy for exactly one run of the same public exact-word task, all roles stock `bonsai-stock`, at most four calls, two candidates and two rounds. Its maximum path is propose/evaluate/review/revise/evaluate/review. If the first candidate passes, completion after its review is allowed; revision is not guaranteed. A passing reviewed revision may complete after four calls. Exhaustion otherwise exits 3. No experimental or research quality claim follows from either outcome.

```sh
flock -n /tmp/cpu-coordinator-6t3yfB/test.lock timeout 380 nice -n 10 python3 -B content_factory.py research-start --spec-file research/bounded-run-spec.json --out .research-bounded-baseline-001 --lease /home/luizh/Documents/Codex/2026-10-08/task-8/content-factory-shared.lease
```

Variable model-generated candidates and opinions are JSON data in the frozen prompt generator. They cannot change the goal, evaluator, model, limits, control state or command; no model tools, shell, network, host actions, messages, deployments or runtime installs are authorized. Assembled prompts and replies are capped at 4096 UTF-8 bytes; model output is capped at 64 tokens. Each actual assembled prompt must pass the approved offline token-fit procedure before its stage. The proposed scope must explicitly authorize this bounded generator and fit procedure, rather than require a human approval of every generated string. This proposal implements no such authority or fit-policy change.

Physical lane: RX 6600 XT via Vulkan0, all tensors Vulkan0, GPU layers 99, context 4096, four CPU threads. Whole run 360s, per-stage maximum 120s, existing host/GPU supervisor limits 60s/50s, outer timeout 380s. Available RAM admission 16GiB, running floor 12GiB, process RSS 12GiB, address space 16GiB, global VRAM 7GiB, idle VRAM maximum 751619276 bytes. Existing swap/pressure, kernel/core, ownership, release and postflight guards remain unchanged. Nonblocking CPU lock and factory lease plus fresh exclusive admission are mandatory.

Execution is still held by require_real_execution(). A supported platform must recognize the precise scope; local files, environment or CLI flags cannot grant authority. After review, a narrowly scoped gate implementation would be needed at RealAgent, LlamaBackend and native_worker while preserving unrelated holds. A reviewed backend seal must cover changed root sources, and an additional reviewed manifest must freeze research package, evaluator, specs and generator identities. Verify pinned model/runtime/template and qualify the new 4096-context workload; earlier welcome-only 512-context qualification does not transfer. This task did not enact policy, refresh seals or execute inference.

Pinned model: prism-ml/Ternary-Bonsai-2-27B-gguf revision b072e1d3b35a0a630cece372c2127528e0994386; weights Ternary-Bonsai-2-27B-PTQ1_0.gguf, 5946648928 bytes, SHA-256 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3. Prism runtime commit adfffbe41b2cabcd51fff326ab045662265062bb; binary SHA-256 c1b84e353ab86f3444603f59c944283b1fe1043359585b8ddb843bd2a10f5686.

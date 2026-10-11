# Paper-inspired local research laboratory

The reusable `research-mode-*` interface supports configurable, finite PMI research runs with the Bonsai profile and the interval-union correctness evaluator. The default two seeds, one revision, 18 workflow model calls, three experiments and 2,400 workflow seconds describe the first experiment; they are not project limits. [RESEARCH-GOAL.json](RESEARCH-GOAL.json) preserves that interval experiment in `goal` and records the broader project intention separately in `project_goal`.

This is an engineering adaptation of [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y) and its [official supplement](https://media.springernature.com/original/springer-static/esm/art:10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf), especially Note 8 (PDF pages 56–61) and Note 9 (62–69, one-based). It does not reproduce the unavailable original implementation or establish the paper's scientific effectiveness. Command execution is this repository's extension; Co-Scientist must not be confused with the distinct Coscientist system.

## Evidence status

The generalized laboratory passed 54 focused model-free tests: 39 existing controller/profile tests and 15 revision/lifecycle tests, plus a focused projection rerun. Fixtures exercise an 18-call/three-experiment default loop and a 45-call/five-experiment loop with three revisions. These are controller and persistence contracts, with no actual model or candidate execution. The installed whole-project suite subsequently passed 433 model-free tests; the focused 54-test result is a subset, not an additional live qualification.

The unchanged authored CPU qualification passed 884/884 frozen cases: [FIRST-QUALIFICATION-VERIFICATION.json](FIRST-QUALIFICATION-VERIFICATION.json) and [FIRST-CORRECTNESS-RECEIPT.json](FIRST-CORRECTNESS-RECEIPT.json). It is not evidence about a generated candidate. Historical model attempt V4 produced one observed native inference completion, zero application-accepted model completions and zero generated-candidate receipts. Its application stopped on postflight coverage validation. The [archived bundle](runtime-evidence/attempt-v4/bundle-manifest.json) and [posthoc validation](runtime-evidence/attempt-v4/posthoc-calibration-validation.json) preserve that distinction. Posthoc inspection did not invoke a model, perform fresh sampling or create a regular application result receipt.

The first reusable-runtime live run is now recorded in [live-v1 verification](runtime-evidence/live-v1/verification.json). Attempt 1 stopped at GPU idle admission on a 12% sample, before native launch. An explicit new attempt after verified preworkflow cleanup accepted one real calibration with `RealTextAndReleasePass`. Its seed-1 workflow load then stopped because host PSI-full growth reached 251,060 microseconds, above the unchanged 250,000-microsecond guard. There were zero completed workflow model calls and zero generated candidates. Independent cleanup was verified; the uncertain reserved workflow task remains held. The frozen 87-file evidence bundle preserves these outcomes. Attribution of host pressure, including any effect of hashing or page cache, remains unproven. A complete live research loop and live committed-boundary pause/resume remain untested.

## Operator workflow and scope

See [RUNTIME.md](RUNTIME.md) for complete arguments and artifact layout. The operator commands are:

```sh
python3 -B content_factory.py research-mode-plan
python3 -B content_factory.py research-mode-start --out /absolute/path/to/new-run
python3 -B content_factory.py research-mode-status --run-dir /absolute/path/to/new-run
python3 -B content_factory.py research-mode-pause --run-dir /absolute/path/to/new-run
python3 -B content_factory.py research-mode-resume --run-dir /absolute/path/to/new-run
python3 -B content_factory.py research-mode-cancel --run-dir /absolute/path/to/new-run
```

Plan validates configuration without loading a model or executing a candidate. New runs accept a bounded goal, goal file or complete specification; a complete specification cannot be mixed with configuration overrides. The supported scientific task remains interval-union correctness even when the operator changes the goal wording. Configuration is immutable within a run; resume cannot change the goal or inflate its budgets.

| Setting | Default | Finite engineering range |
| --- | ---: | ---: |
| Initial candidates | 2 | 2–8 |
| Evolutionary revisions | 1 | 0–16 |
| Workflow model calls | 18 | 1–4,096 |
| Experiment calls | 3 | 1–24 |
| Workflow wall seconds | 2,400 | 1–604,800 |

Model/calibration stages remain capped at 120 seconds and experiment stages at 60 seconds. Each explicit attempt has one separate calibration allowance and 180 seconds reserved for cleanup. Resume preserves cumulative workflow consumption. Additional candidates or revisions do not silently increase budgets: compare the plan's expected full-population model-call and distinct-candidate-check counts with the configured limits. Partial budgets can finish with `budget_exhausted`; configured scope is not execution authority.

The runtime uses normal platform review, exact process-local operation authority, source/runtime/model identity checks and fresh per-stage resource admission. Arbitrary shell commands, other models, benchmark evaluators and network retrieval are unsupported. Generic factory paths, historical one-shot reservations and unrelated holds remain unchanged. The generic `research-lab-*` start/resume route directs this profile to `research-mode-*`.

## Research loop and local approximations

A durable serial queue generates candidates, executes their independent checks, reflects on the receipts, constructs a proximity graph, records pairwise debates and final judgments, updates Elo, synthesizes feedback, evolves and checks successive revisions, and produces an overview. Generation, reflection, ranking, evolution, proximity and meta-review correspond to the paper's six roles. Debate and overview are additional queue stages; proximity is calculated locally. Finite waves and FIFO/evidence-first/adaptive queue priorities approximate the paper's asynchronous workers, without a parallel worker pool.

Candidates, source hashes, parent IDs and completed results are immutable. Seed prompts carry persisted task identities and different requested algorithm approaches; later generation sees prior source evidence. Different requests do not prove actual algorithmic diversity. Exact duplicates link to their original source and skip redundant checks and matches. Reflection waits for measured evidence, including the original receipt for a duplicate. Every distinct new revision is reviewed and compared with prior distinct candidates. The measured incumbent remains separate from model rank: an incorrect high-Elo candidate cannot replace a correctness-passing incumbent.

Generation/evolution styles, K16/32, one to five total turns per comparison, one or two comparisons per pair, presentation order and match scheduling are local bounded choices. Defaults use K32 and two debate calls followed by one ranking call per pair. Stable evolution requires at least two completed comparisons of each nonduplicate pair and sufficiently small latest observed Elo updates reconstructed from persisted matches. Distance from the initial rating of 1200 is not a stability measure. These thresholds and coverage rules are an inferred convergence proxy, requiring equal-budget and reversed-order ablations before usefulness claims.

The research profile accepts token-Jaccard or character-trigram lexical proximity. It rejects model-semantic mode; lexical overlap is not the paper's embedding implementation or a validated measure of algorithmic similarity. Operator excerpts retain origin, URL and content hashes. Remote retrieval, multiple independent verification reviewers, continuous asynchronous allocation and within-run goal revision remain unimplemented. [PROVENANCE-MATRIX.json](PROVENANCE-MATRIX.json) records each adaptation, uncertainty and invalidation test.

## Prompt, response and evidence contracts

Every model task persists its complete input snapshot and SHA-256, exact rendered prompt, response identity and applicable profile/fit metadata. Source and evaluator receipts remain complete in durable state. Aggregate prompts show at most three complete candidate rows: for larger populations, `latest-with-parents-v1` selects the newest candidate and its parents, with top-Elo fallback. The payload explicitly lists selected and omitted IDs/counts and the hash resolving the full persisted snapshot. Relevant reviews, debate turns and decisions remain whole. Generation includes the last two prior sources and source-hash references for earlier candidates. This bounded context selection can omit useful older evidence; it is not lossless model access to the whole population.

The profile uses `bonsai-stock`, 4,096 context tokens, 512 output tokens for calibration/generation/evolution and 128 for other model roles. JSON-serialized string-content caps are 640 bytes for code, 192 for reflection/debate/meta-review, 96 for ranking rationale and 512 for overview. Source syntax is checked without executing it. The provisional single-function subset intentionally forbids annotations, including `AnnAssign`, beyond the broader original validator. Tuple inputs, canonical list-of-list outputs, permitted builtins and other parser rules are explicit coding constraints, not paper requirements.

Strings are not silently truncated or repaired. Exceeding a response cap, syntax rule or prompt envelope blocks the task. Larger goals, populations or debate settings may still overflow despite projection. The default all-cap fixture's largest prompt is 3,798 bytes; the conservative arithmetic is 3,798 + 136 wrapper/boundary/slack + 128 output reserve = 4,062 of 4,096. That fixture calculation is not an exact tokenizer count or actual fit receipt. Every physical prompt must pass the runtime's conservative rendered-byte fit check with its own role output reserve. Generic 64-token settings remain unchanged.

## Persistence and lifecycle

One immutable run configuration and one SQLite workflow span numbered attempts. Tasks are reserved before external work. Complete task IDs, candidates, receipts, prompt snapshots and ancestry survive reopen/resume. A reserved task without a committed result is uncertain and cannot be automatically replayed; neither model output nor a fabricated receipt can resolve it.

Pause is checked only between committed tasks with no in-flight reservation. The workflow remains `ready`, with a paused child result. Resume requires the previous supervisor's independently verified cleanup, unchanged configuration and remaining budgets; it creates a new attempt and separate calibration without rerunning completed tasks. This is a task-boundary checkpoint, not a suspended model process. Cancellation is durable and permanent.

If calibration ended before a workflow database existed, explicit resume may create a new attempt only after verified cleanup; it uses the same frozen run configuration. An existing `blocked`, `finished`, `cancelled` or `budget_exhausted` workflow is not treated as a calibration-only failure. Missing cleanup or uncertain work prevents continuation. Model-free lifecycle tests do not prove physical pause/resume behavior on this host. Live-v1 does establish an explicit new attempt after a clean preworkflow admission stop; it does not establish continuation of an existing paused workflow.

## Correctness and interpretation

The interval experiment implements `merge_intervals(intervals)` for integer half-open intervals without mutating tuple inputs. An independent endpoint-sweep oracle checks 884 frozen exhaustive, edge and seeded cases outside the candidate sandbox. Correctness receipts have null timing fields. Passing these cases does not establish universal correctness, novelty, performance improvement or research effectiveness.

Public results distinguish workflow termination, accepted native/application operations, actual trusted correctness success and complete comparative coverage. Expected coverage changes with the configured population and comparison options. Duplicate or partial populations may finish without complete distinct-candidate coverage. Fixtures never establish actual objective success, and high model ratings do not establish correctness.

Feedback-on/off, seed-diversity, reversed-order, strategy, projection-window and direct-generation comparisons under equal live budgets remain unrun. Performance experiments require a separate reviewed measurement scope and held-out profiles. The historical `end-to-end-proposal.json`, fixed `model-demo-spec.json`, failed/consumed attempts and authored CPU evidence remain preserved. [EXECUTOR.md](EXECUTOR.md) records the scoped execution controls, historical outcomes and remaining physical qualifications.

### Direct integrity-read qualification

The research profile now hashes the complete GGUF through a bounded direct-I/O reader, avoiding application-buffered model reads. Thirteen focused tests pass, including unaligned EOF and failure handling. The guarded full-model checksum was refused before dispatch because host RAM was below the unchanged 16 GiB admission floor; it remains unqualified on the full model. This is a memory-footprint mitigation, not proven attribution or a demonstrated cure for the prior host-pressure stop. See [frozen verification](runtime-evidence/direct-hash-v1/verification.json). No cache eviction, machine setting, or guard threshold changed.

Current integrated verification: **446 model-free tests pass**. See [the runtime verification report](RESEARCH-RUNTIME-VERIFICATION.json) for separate passed, failed and not-run checks, preserved attempts, and the next delivery step. The complete live discovery loop is still blocked by host resource admission.

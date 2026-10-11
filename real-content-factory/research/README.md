# PMI research mode

The operator supplies a goal, catalog model IDs, immutable executable checks and finite budgets. PMI serially proposes a candidate, evaluates it in trusted code, stores a model critique, and revises the best eligible candidate. Reviews are opinions and cannot change checks, permissions, completion or rankings. The earliest candidate wins score ties. Failed safety checks cannot become best.

Inspect an offline plan:

```sh
python3 -B content_factory.py research-plan --spec-file research/example-spec.json
```

Create a durable run using the existing shared resource lease:

```sh
python3 -B content_factory.py research-start --spec-file research/example-spec.json --out /path/to/new-research-run --lease /path/to/existing-shared.lease
python3 -B content_factory.py research-status --run-dir /path/to/new-research-run
python3 -B content_factory.py research-resume --run-dir /path/to/new-research-run --lease /path/to/existing-shared.lease
python3 -B content_factory.py research-cancel --run-dir /path/to/new-research-run
```

The supplied specification is trusted operator input. `text-checks-v1` supports required/forbidden terms and a UTF-8 byte limit; `exact-text-v1` compares the entire string to immutable `expected_text` without trimming, case folding or normalization and also enforces a byte limit, with no shell, dynamic evaluator code, deployment or messages. Passing these checks means completion of the declared text contract, not experimental verification or scientific discovery. Unsupported experimental evaluators report blocked. Prompts and replies are bounded at 4096 UTF-8 bytes; oversized assembled prompts stop rather than silently dropping constraints. Existing 64-token text-generation limits are not raised.

Each run stores a versioned SQLite database with the immutable goal/config/evaluator identity, candidates, evaluations, opinions, best ID, model-stage reservations and budgets. SQLite commits use FULL synchronization; run and shared-lease locks prevent competing runners. Status and cancellation use brief transactions and do not construct adapters. No fixture backend is exposed by the production CLI. Exit codes: complete=0, blocked=2, budget_exhausted=3, cancelled=130; planning/status/cancel control return 0.

Resume continues only committed phase boundaries, without repeating completed model calls. Unreconciled in-flight work stays blocked; there is no automatic retry, reconciliation override or fixture-to-real promotion. Budgets survive resume. A cancellation request during a stage is persisted immediately and honored at the next verified-cleanup boundary; it does not promise immediate termination. Model calls remain bounded by the existing adapter deadline and ownership supervisors. Abrupt interruption without release proof remains blocked, even after cancellation.

Live research is currently held by `require_real_execution()` before model-stage creation or runtime import. The durable goal and blocker are still saved. The exact Kokoro speech exception provides no research authority. New research prompts need their own token-fit/model qualification and supported execution approval, plus fresh admission. This implementation did not run local inference or demonstrate discovery.

This source increment changes `content_factory.py` and adds `test_research.py`, invalidating the prior Kokoro source seal for any new synthesis. Preserve the successful speech sample and receipts; refresh the seal through normal review before a future authorized speech run. Image/video/text/audio adapters and physical gates themselves are unchanged.

Model-free regression command, only in an assigned CPU interval:

```sh
flock -n /tmp/cpu-coordinator-6t3yfB/test.lock python3 -B -m unittest -q test_contract test_speech test_research
```

The discovery-loop shape is inspired by [Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y). It does not reproduce its experimental work, multi-agent scale or training.

Completion reports only the declared evaluator checks. Goal semantics beyond those checks are not independently verified. The exact-word smoke and inactive bounded policy are described in [LIVE-SMOKE.md](LIVE-SMOKE.md); neither demonstrates research quality or grants execution authority.

The explicit population/tournament laboratory is available through `research-lab-*`; see [LABORATORY.md](LABORATORY.md) and [PROVENANCE-MATRIX.json](PROVENANCE-MATRIX.json). Candidate execution remains fail closed. The previous exact-word live proposal is held and is not the proposed research demonstration.

# Reusable research runtime

`research-mode-*` is the operator entry for persistent PMI research runs. Each run
keeps its goal, validated configuration, task reservations, model evidence,
candidate sources, evaluator receipts and ancestry. The supervisor runs one stage
at a time and requires owned cleanup before the next stage.

This is a bounded engineering adaptation of the [Nature paper](https://www.nature.com/articles/s41586-026-10644-y)
and its [official supplement](https://media.springernature.com/original/springer-static/esm/art:10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf).
Generation, reflection, ranking, evolution and meta-review use the selected
research profile; proximity is a local lexical calculation. Debate and overview
are additional queue stages. A finished queue, a passing bounded correctness
check and a complete comparative demonstration are separate results.

## Plan and start

Run these commands from the project directory. Planning validates inputs and
prints the proposed configuration without loading a model or executing a
candidate:

```sh
python3 -B content_factory.py research-mode-plan \
  --goal 'Develop a correct merge_intervals implementation and inspect its counterexamples.'
```

For a reviewed live launch, choose a new absolute output directory:

```sh
python3 -B content_factory.py research-mode-start \
  --goal-file /absolute/path/research-goal.txt \
  --out /absolute/path/research-runs/intervals-001
```

`--goal` supplies text directly; `--goal-file` reads the text from a bounded file.
`--spec-file` supplies a complete JSON configuration. Use one goal/configuration
input at a time. The configuration is frozen for the run; resume reads it from
the run directory rather than accepting replacement goals or increased budgets.
With none of these inputs, the default goal comes from `RESEARCH-GOAL.json`.
The JSON from `plan` contains the specification under `config`; pass that object,
not the entire plan report, to `--spec-file`.

Without `--spec-file`, these flags override the defaults for a new plan or run:

| Flag | Default | Accepted range |
| --- | ---: | ---: |
| `--initial-candidates` | 2 | 2–8 |
| `--evolutions` | 1 | 0–16 |
| `--model-calls` | 18 | 1–4,096 |
| `--experiment-calls` | 3 | 1–24 |
| `--wall-seconds` | 2,400 | 1–604,800 |

For example, inspect a larger finite proposal before requesting its launch:

```sh
python3 -B content_factory.py research-mode-plan \
  --goal-file /absolute/path/research-goal.txt \
  --initial-candidates 4 --evolutions 3 \
  --model-calls 120 --experiment-calls 7 --wall-seconds 7200
```

A complete specification also controls the supported scheduling, generation,
comparison and lexical-proximity options and bounded operator evidence. It
cannot be combined with CLI configuration overrides. More candidates or revisions
do not silently increase budgets. Compare `expected_full_population_model_calls`
and `expected_distinct_candidate_checks` with the configured limits; a smaller
budget intentionally allows only partial coverage and may end in
`budget_exhausted`.

The default configuration has two seeds, one evolution, eighteen workflow model
calls, three experiment calls and 2,400 seconds of workflow time. Each attempt
first runs one separate calibration call. Model/calibration stages have a
120-second cap; experiment stages have a 60-second cap. The outer supervisor
reserves another 180 seconds for cleanup. The model and experiment stage guards
cannot be expanded through user configuration.

Planning and recording an integrity seal do not approve physical execution.
The launch uses the normal platform review and the existing model fit,
host-resource admission, ownership and cleanup checks. It does not launch an
additional Codex chat or change the Codex model, configuration or permissions.

## Observe, pause, resume and cancel

```sh
python3 -B content_factory.py research-mode-status --run-dir /absolute/path/research-runs/intervals-001
python3 -B content_factory.py research-mode-pause --run-dir /absolute/path/research-runs/intervals-001
python3 -B content_factory.py research-mode-resume --run-dir /absolute/path/research-runs/intervals-001
python3 -B content_factory.py research-mode-cancel --run-dir /absolute/path/research-runs/intervals-001
```

The module interface accepts the same actions without the prefix, for example
`python3 -B -m research.runtime_entry status --run-dir /absolute/path/run`.

Status is a read-only snapshot. Pause requests a stop between committed tasks;
it does not suspend a model process halfway through a stage. Resume requires the
preceding supervisor to have completed verified cleanup, preserves committed
results and remaining budgets, and creates a new numbered attempt. Only the
pause marker is removable through resume. Cancellation is durable and is also
recorded in the workflow database when it exists; it cannot be undone by resume.

If calibration stopped before the workflow database was created, a new explicit
resume is allowed only after the previous supervisor proves cleanup. It performs
a new calibration and then creates the workflow with the same configuration.
An existing workflow marked `blocked`, `finished`, `cancelled` or
`budget_exhausted` is retained for inspection and cannot be resumed. Paused
workflows retain a `ready` database state and a `paused` child result.

A reserved task without a committed result is uncertain work. It stays held
after a crash or stage failure; resume does not rerun it or synthesize a receipt.
Missing cleanup evidence also prevents another attempt. Preserve the directory
and inspect the recorded reason rather than deleting receipts, resetting counters
or editing the database to force continuation.

## Stored artifacts

```text
run/
  run.json                         immutable configuration and identity
  launch.lock                      excludes overlapping launchers
  pause.json                       optional durable pause request
  cancel.json                      optional permanent cancellation request
  workflow/
    laboratory.sqlite3             persistent tasks, candidates and evidence
    task-*/                        prompt, response or experiment artifacts
  attempts/
    0001/
      launch.json                  immutable scope for this attempt
      authority-reserved.json      consumed launch identity
      calibration/                 per-attempt calibration evidence
      child-result.json            child workflow result
      supervisor-report.json       lifecycle and cleanup result
      launch-error.json            present if launch authority/supervisor fails
      controller.stdout.log
      controller.stderr.log
    0002/                          later authorized continuation, when eligible
```

The workflow stays in one location across attempts. Completed task IDs, exact
prompt snapshots, response identities, candidate source hashes and parent IDs
are preserved. The run-level launch lock prevents overlapping attempts for the
same run, and the shared resource coordinator controls physical stage ownership.
Inspect the child result and workflow summary along with the supervisor report.
The supervisor's `completed` flag concerns its protocol, and successful process
exit alone does not establish `objective_met` or full demonstration coverage.

## Scope and interpretation

The currently supported backend is `bonsai-stock` with the
`bonsai-research-v1` profile and the immutable interval-union correctness
evaluator. Arbitrary shell commands, candidate entry points, models, benchmark
evaluators and network retrieval are not accepted through this interface.
Operator-provided evidence must retain its provenance and content hash.

The generated algorithm is evaluated separately from the model's reflection,
debate, ranking and Elo opinions. A candidate does not become objectively correct
because it receives a high model rating. The CPU evaluator has 884 frozen cases;
its per-invocation limits remain 10 seconds wall, 5 seconds CPU and 512 MiB
address space, with a 256 MiB aggregate RSS stop. The longer experiment-stage
allowance covers admission and controller work and does not relax these limits.

Contract tests use fixtures and forbid physical process launches. Passing them
does not establish actual model inference, candidate execution, scientific
novelty, paper reproduction, performance improvement or the efficacy of the
multi-role design. See [LABORATORY.md](LABORATORY.md),
[EXECUTOR.md](EXECUTOR.md) and [PROVENANCE-MATRIX.json](PROVENANCE-MATRIX.json)
for the recorded approximations and outstanding validation work.

### Direct integrity-read qualification

The research profile now hashes the complete GGUF through a bounded direct-I/O reader, avoiding application-buffered model reads. Thirteen focused tests pass, including unaligned EOF and failure handling. The guarded full-model checksum was refused before dispatch because host RAM was below the unchanged 16 GiB admission floor; it remains unqualified on the full model. This is a memory-footprint mitigation, not proven attribution or a demonstrated cure for the prior host-pressure stop. See [frozen verification](runtime-evidence/direct-hash-v1/verification.json). No cache eviction, machine setting, or guard threshold changed.

Current integrated verification: **446 model-free tests pass**. See [the runtime verification report](RESEARCH-RUNTIME-VERIFICATION.json) for separate passed, failed and not-run checks, preserved attempts, and the next delivery step. The complete live discovery loop is still blocked by host resource admission.

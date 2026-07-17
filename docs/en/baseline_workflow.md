# Reproducible Baseline workflow

## Scope

A Baseline is an immutable, versioned definition of an experiment. It is not a
copy of LeRobot data and it never contains policy credentials. The backend
filesystem store is authoritative; browser storage is not used.

Each Baseline records Git commit, policy URL/label/metadata, optional checkpoint
hash, ActionSpec, state schema, camera roles/configuration, robot/runtime/safety
settings, RTC and warmup settings, evaluation protocol, start-position protocol
and optional source dataset/episode/sample references. Missing checkpoint hashes
are saved as `null`, never fabricated.

## Create, derive, run

1. Connect the intended DEPLOYMENT runtime and save its configuration.
2. In the Web **Baseline** page, create a Baseline with policy label, task,
   protocol and operator metadata. The bounded writer persists it atomically.
   An existing current `EvaluationSession` can instead be captured with **从当前评测创建**
   or `mp-baseline create --from-evaluation-snapshot snapshot.json`.
3. For a changed checkpoint, RTC setting, camera, FPS, safety limit or start
   state, clone the old Baseline and state the derivation reason. Keep one major
   variable per formal A/B comparison where practical.
4. **从 Baseline 创建评测** compares the live configuration with the Baseline.
   Any execution-relevant difference is rejected and displayed by category.
5. The operation only creates an `EvaluationSession`; an operator must still
   warm up, acknowledge reset, and start each episode manually.
6. Terminal Baseline-backed evaluations are attached by a background writer.
   Use `attach-open-loop` or the Web association control for an open-loop result.

## Result semantics

Success rate is always displayed as percentage, numerator, denominator and
sample size. `SUCCESS`, `FAILURE`, `TIMEOUT`, and `SAFETY_ABORT` are valid
trials. `INVALID`, `SYSTEM_ERROR`, and `OPERATOR_ABORT` remain visible but are
excluded from the success-rate denominator. A 1/1 result is displayed as
`100% (1/1; valid n=1)` and is a smoke result, not a stable conclusion.

Results with differing robot, dataset format, ActionSpec, state schema, or
camera roles are not pooled. Open-loop comparisons also require matching target
source and alignment mode.

## CLI

`mp-baseline create` accepts metadata JSON plus a sanitized runtime-config JSON;
the Web page is simpler for live deployments. `run --web-url` calls the Web
Baseline run endpoint and creates a manual session only. It never starts robot
motion by itself.

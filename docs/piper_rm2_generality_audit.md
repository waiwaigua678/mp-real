# Piper / RM2 Generality Audit

## Scope

The H6 audit checked shared runtime, data, evaluation, pose, replay and Web
wiring for fixed Piper layout assumptions: action dimension 14, fixed three
cameras, fixed gripper indices, direct vendor SDK imports and Piper-only
metadata.

## Findings

| Area | Result | Evidence |
| --- | --- | --- |
| Runtime loops | pass | Shared sync/RTC/infer-only code consumes `Robot`, `ActionSpec` and camera roles. |
| Recording metadata | H6 updated | New metadata is derived from `RecorderConfig.action_spec`. |
| Replay planning | H6 updated | Valid dual-arm specs preserve `arm_count=2`; unknown layout remains invalid rather than becoming a fake single arm. |
| Replay feedback | pass | H5 separates sent, feedback, acknowledged and displayed cursors. |
| Open-loop evaluation | H6 updated | ActionSpec snapshots now use `ActionSpec.to_dict()`. |
| Core dependencies | H6 updated | `av` and `pyarrow` moved to optional data/recording/evaluation extras. |
| Vendor SDK imports | pass | Vendor SDK calls remain under `robots/piper/` and `robots/rm2/`; offline/data commands do not import vendor SDKs. |
| Web configuration | partial | `web/server.py` still has compatibility Piper-shaped CLI fields while selecting robot-owned `RobotWebProfile` implementations. |

## Current Boundaries

- Shared modules must not hard-code Piper's 14-dimensional layout.
- Shared modules must not assume fixed camera roles or gripper dimensions.
- Robot-specific SDK calls stay below `src/mp_real/robots/piper/` and
  `src/mp_real/robots/rm2/`.
- New metadata must round-trip through `ActionSpec`.

The remaining Web compatibility fields are migration debt and should be handled
in a separate refactor, not during H6 hardening.


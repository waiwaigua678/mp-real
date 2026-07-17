# Stage 6 hardware validation: LeRobot v2.1 recording

This document is a runbook, not authorization to move hardware.  Each
motion-capable level requires a separate explicit approval in the active task.
Keep the physical emergency stop available throughout and preserve the prior
deployment revision until the resulting datasets validate.

## Before every level

- Create a new dataset root; never append to a finalized recording.
- Record the robot, camera roles, policy endpoint, operator, task and intended
  level in the run log.
- Confirm `mp-data-inspect <dataset>` and `mp-data-validate <dataset>` are
  available locally.
- Stop if the policy reports a safety rejection, the recorder reports an error,
  camera age exceeds the configured threshold, the writer queue grows without
  draining, or any operator observes unexpected motion.

## L1 — fake resources only

Run the unittest suite with fake robot, camera and policy implementations.
Expected result: a two-episode v2.1 dataset validates, videos align with
Parquet rows, and an external standard v2.1 dataset can be read.

```bash
uv run python -m unittest tests.test_lerobot_v21 tests.test_evaluation -v
uv run mp-data-validate <generated-dataset>
```

No hardware command is issued at this level.

## L2 — live cameras and robot state, no commands

Requires explicit approval. This stage needs a deliberately configured
no-command recorder path; Stage 6 does not automatically turn infer-only
execution into a recording session. Record 60 seconds of stationary
observations, then stop normally.

Expected result: every camera has exactly one MP4 frame per Parquet row;
`camera_frame_reused` and `camera_age_ns` explain repeated frames; no CAN or
vendor command log contains an action command. Collect controller logs, writer
queue high-water mark, dropped counters, `meta/mp_real/events`, and validator
output. Roll back by stopping the deployment, closing cameras, and deleting
only the newly created unfinalized dataset directory.

## L3 — low-speed unloaded movement

Requires explicit approval after stating the exact policy, expected workspace,
speed limits and emergency-stop procedure. Use the existing conservative robot
limits; do not raise speed for recording. Record 30 seconds with no payload.

Expected result: the standard `action` equals the executed action reported by
the robot boundary; raw and stabilized values remain in mp-real telemetry; the
writer queue stays bounded and control-period telemetry remains within the
deployment's existing safety expectations. Validate immediately after the run.

Stop by using the normal Web stop control or emergency stop. If any mismatch is
found, disable the deployment, retain the `.inprogress` directory and logs for
diagnosis, and return to the prior runtime revision.

## L4 — production episodes

Requires explicit approval for each robot and each motion session. Record at
least three Piper and three RM2 episodes, including a normal completion and a
manual abort for each robot.

For each finalized dataset run:

```bash
uv run mp-data-inspect <dataset>
uv run mp-data-validate <dataset>
```

Archive validation output, session/label metadata, events, telemetry and any
recovery metadata. A failed or aborted run must remain clearly marked
`INCOMPLETE` or `INVALID`; do not rename it as a valid training dataset.

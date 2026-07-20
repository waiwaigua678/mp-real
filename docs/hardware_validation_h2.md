# Hardware Validation H2

H2 changes recorder memory behavior only. It does not change replay safety,
move-to-state safety, robot action semantics, joint limits or policy behavior.

No hardware validation is required for automated tests. Do not run any command
that can move, enable, reset or command a robot unless the operator explicitly
authorizes that motion test for the current task.

## Static Recording Check

Purpose: verify recorder memory, queue and cache metrics with real cameras
while the robot remains stationary.

1. Start an evaluation with `save_data=true`.
2. Use three configured cameras.
3. Record for 15 minutes without sending robot motion.
4. Monitor RSS, `recording.metrics.queue_size`,
   `recording.metrics.cache_entry_count`,
   `recording.metrics.buffered_image_bytes`,
   `recording.metrics.buffered_telemetry_bytes`,
   `recording.metrics.telemetry_part_count`,
   `recording.metrics.dropped_frame_count` and
   `recording.metrics.dropped_event_count`.
5. Stop and label the episode.
6. Confirm no recorder, encoder or camera-preview worker thread remains alive.

Expected result: queue and cache metrics stay bounded, telemetry parts increase
over time, and dropped counters remain zero unless an intentional overload test
is being run.

## Low-Speed Empty-Space Recording

This check can move the robot and requires explicit approval.

Before running it, state:

- the exact command or Web workflow
- why motion can occur
- expected low-speed motion
- stop procedure
- rollback procedure

Then wait for the operator to approve with the current-task phrase for real
robot motion testing.

Validation steps after approval:

1. Record a 5 minute low-speed empty-space episode.
2. Confirm standard LeRobot rows remain control-step aligned.
3. Confirm telemetry parts and metrics stay bounded.
4. Stop normally and verify writer/encoder/file handles close.

## Repeated Episode Check

Purpose: detect old cache retention across episodes.

1. Record 10 short episodes.
2. After each label/save, inspect `cache_entry_count` and queue depth.
3. Confirm old episode image references are not retained.
4. Confirm final dataset validation passes.

## Abort Check

1. Start recording.
2. Stop or abort before normal finalization.
3. Confirm `.inprogress` remains.
4. Confirm `meta/mp_real/recovery.json` exists.
5. Confirm completed telemetry parts for written frames are readable.
6. Confirm all recorder-owned worker threads exited.

## No-Go Conditions

- Any queue/cache metric grows linearly with wall time after the writer has
  caught up.
- Dropped frame/event counters increase without operator-visible reporting.
- Incomplete datasets contain format-valid rows whose telemetry parts are
  missing for those rows.
- Writer failure does not move the evaluation session to an error/abort path.
- Any encoder or file descriptor remains open after finalize or abort.

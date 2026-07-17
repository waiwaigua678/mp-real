# Stage 7 hardware validation: Web evaluation workflow

This checklist is for an operator performing the approved real-hardware
validation after the automated fake-resource tests pass.  It is not a command
to move either robot automatically.

## Record before testing

- Git commit:
- Robot: Piper / RM2
- Policy URL and policy label:
- Dataset root:
- Operator:
- Runtime configuration snapshot:
- ActionSpec and camera roles:

## Preconditions

1. Clear the workspace and confirm the physical emergency-stop procedure.
2. Verify the configured speed, safety limits, and reset pose are appropriate
   for this task.  Do not increase them for this validation.
3. Open the Web UI, select `DEPLOYMENT`, connect resources, and verify camera
   preview.  Do not start ordinary deployment control.
4. Create an evaluation with 3–5 planned episodes and record the generated
   session id and dataset path.

## Per-robot procedure

Perform these items separately for Piper and RM2.

| Check | Expected evidence |
| --- | --- |
| Connect without motion | Cameras and policy metadata appear; no action loop is running. |
| Warmup | State reaches `WAITING_RESET`; no robot action occurs and no formal episode is created. |
| Manual-stop episode | `RUNNING` → `STOPPING` → `WAITING_RESULT`; control thread has exited before labels are enabled. |
| Timeout episode | Timeout trigger is displayed, then label as failure or invalid according to operator judgement. |
| Invalid episode | Label `INVALID`; success rate denominator does not increase. |
| Normal task episode | Label success or failure with a reason; dataset episode and metadata are written. |
| Page refresh | State, current dataset path, legal actions, and label controls restore from the backend. |
| Double-click | A repeated start/label returns a clear conflict and does not create another action loop or label. |
| Brief network interruption | Browser reconnects by polling; backend state remains authoritative. |
| Completion | Dataset is finalized, `episode_labels.jsonl` exists, and all evaluation workers have exited. |

## Record after every episode

- Episode index, session/generation id, result, failure reason, and stop trigger.
- Recorder queue depth/high watermark and dropped event/frame counts.
- Control FPS/period and policy timing displayed by the UI.
- Any original exception type/message and corresponding server log excerpt.

## Stop and rollback

If an unexpected motion, safety stop, policy failure, or recorder failure
occurs, use the physical stop procedure first.  Then use **中止评测** in the
Web UI, wait for the controller and recorder workers to exit, preserve the
`.inprogress` directory and logs, and do not retry the same motion until the
failure has been reviewed.

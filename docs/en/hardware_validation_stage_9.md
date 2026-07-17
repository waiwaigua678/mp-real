# Stage 9 hardware validation: move to a recorded robot state

This is an operator checklist, not permission for an automated motion test.
Run every gate independently for Piper and RM2. Passing a gate for one robot
does not pass it for the other.

## Preconditions

- Record Git commit, robot, operator, target dataset/episode/sample, mapping
  config fingerprint, ActionSpec snapshot and plan hash.
- Clear the workspace, verify the physical stop procedure, and keep the
  configured pose speed at or below the conservative configured value.
- Confirm vendor stop, health, joint-limit and workspace validation are
  available. The current implementation rejects a real move when any of
  those capability checks is unavailable; do not bypass that rejection.
- Never use an action, raw action, expert action, or recorded image as a
  target state or live deployment observation.

## Gate sequence

| Gate | Procedure | Required result before continuing |
| --- | --- | --- |
| 0 | Run `mp-move-to-recorded-state` without `--execute`; inspect schema, units, target values and mapping report. | No Robot is created; target derives from `observation.state`; report is valid. |
| 1 | Use Web **连接机器人用于姿态移动** only. | Robot feedback is readable; no enable, reset, camera, Policy, or motion occurs. |
| 2 | Choose a sample effectively equal to the live pose and generate the live plan. Do not execute. | Current state/deltas/units/plan hash are visible and expected. |
| 3 | Execute a minimal joint delta at low speed with workspace clear. | Progress reports stay within tracking limits; physical stop and Web stop both abort safely. |
| 4 | Move to a nearby recorded state, without policy handoff. | Final verification is `reached` (or explicitly reviewed warning). |
| 5 | From a reached state, choose **连接相机和 Policy**. | No reset; warmup actions are discarded; no control loop is running. |
| 6 | Choose **二次确认开始推理** with `max_steps=1`. | The first executed action is from a fresh real camera/state observation, not a recorded frame. |
| 7 | Run a brief low-speed deployment. | Recording metadata includes source sample, `move_plan_id`, and target tracking error. |

## Stop and rollback

At any unexpected motion, health failure, tracking-error breach, control
overrun, communication error, or stale-plan rejection:

1. Use the physical emergency-stop procedure first.
2. Use **停止姿态移动** or stop the deployment controller; wait for worker join.
3. Disconnect, preserve logs and any `.inprogress` recording, and record the
   plan hash and original error message.
4. Do not advance to a later gate until the cause is reviewed.

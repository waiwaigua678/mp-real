# Piper / RM2 generality audit

## Scope and method

This Stage-12 audit searched non-test shared Python modules for fixed action or
state dimension `14`, a `6 + 6` layout, fixed camera role names, fixed gripper
indexes, Piper robot-name tests, CAN fields, and direct Piper/SDK imports. It
also traced resource ownership through `ResourceLeaseManager` and the Web
runtime's deployment, evaluation, pose and replay paths.

## Result

| Area | Result | Evidence / disposition |
| --- | --- | --- |
| `runtime/`, `data/`, `evaluation/`, `common/` | pass | No fixed 14-dimensional or Piper SDK assumptions. Baseline uses `ActionSpec`, state names and camera roles. |
| Baseline configuration capture | pass | Shared Baseline code consumes generic `camera_config`, `robot_config` and `safety_config` sections. Piper/RM2 construct those sections in their own `RobotWebProfile`. |
| Action/state schema | pass | `ActionSpec` round-trips schema version, names, action mode, arm count, gripper indices and capabilities without changing positional construction. |
| Camera/action interpretation | pass | Shared replay, open-loop and recording code follows `ActionSpec` fields and camera roles. Piper/RM2 decoding remains under their robot modules/profiles. |
| Direct vendor SDK imports outside robot packages | pass for SDK imports | No direct `pyagx`/RM SDK import is used by shared runtime/data/evaluation code. Web/profile imports of robot `infer` modules are existing construction wiring, not SDK calls. |
| Shared Web lifecycle | partial | `RobotWebRuntime` selects a profile and works for both robots, but `web/server.py` still retains Piper CLI fields, placeholder-camera helpers and a compatibility legacy RM2 Web implementation. These are documented migration debt, not copied into new Baseline code. |
| Resource ownership | pass in-process | Fake tests cover deployment/evaluation/pose/replay control conflicts, duplicate control/replay leases, stale lease protection, and offline-data plus camera-preview coexistence. Baseline browsing creates no resource lease. |

## Required boundaries

- Vendor SDK calls remain under `robots/piper/` and `robots/rm2/`.
- Shared code must consume only `Robot`, `ActionSpec`, state schema, camera
  roles, profile capabilities and explicit mappings.
- A new robot must supply its own `RobotWebProfile` baseline-category callback;
  it must not add vendor field names to `evaluation/baseline/`.
- The in-process resource manager does not coordinate multiple processes.

## Remaining migration items

The legacy Piper-shaped Web CLI/configuration code is intentionally untouched
in this stage to preserve existing deployment behavior. A later, separately
reviewed refactor can move those flags and camera placeholder helpers fully
behind profile-owned interfaces. It must preserve the current command names and
must be validated with both fake robots before hardware testing.

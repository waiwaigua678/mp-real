# Piper / RM2 capability matrix

`supported` means a software path exists and has fake/mock coverage; it does
not imply hardware validation. RM2 CLI deployment defaults have a separately
recorded operator validation scope. It does not extend to RM2 Web deployment,
recorded-state movement, or trajectory replay. Piper hardware movement/replay
validation is not claimed.

| Capability | Piper | RM2 | Status detail |
| --- | --- | --- | --- |
| CLI deployment | supported | supported | shared sync/RTC/infer-only loops; RM2 defaults have a limited recorded hardware-validation scope |
| Web deployment | supported | supported | shared `RobotWebRuntime` profile path |
| CAMERA_PREVIEW | supported | supported | no Robot or PolicyClient |
| Web DATA_VIEW | supported | supported | recorded-data browser; robot replay remains safety-gated |
| Web policy warmup | supported | supported | warmup actions discarded |
| CLI startup warmup | unsupported | unsupported | CLI follows its existing direct loop path |
| sync / RTC | supported | supported | ActionSpec-driven |
| EvaluationSession | supported | supported | fake robot coverage |
| LeRobot v2.1 recording | supported | supported | bounded writer coverage |
| Offline data view | supported | supported | no hardware resource |
| Move to recorded state | experimental | experimental | physical execution blocked by default safety validation |
| Deployment from recorded state | experimental | experimental | requires verified pose handoff |
| Trajectory replay | experimental | experimental | physical execution blocked pending safety validation |
| Open-loop evaluation | supported | supported | robot-neutral, never commands action |
| Baseline and A/B report | supported | supported | generic ActionSpec and runtime snapshots |

Shared `runtime/`, `data/`, `evaluation/`, pose validation, replay planning and
open-loop metrics use `Robot`, `ActionSpec`, state fields and camera roles.

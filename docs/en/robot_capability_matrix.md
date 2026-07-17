# Piper / RM2 capability matrix

`supported` means a software path exists and has fake/mock coverage. It does
not mean hardware validation. No item below is currently marked
`hardware-validated` without an independently recorded hardware gate.

| Capability | Piper | RM2 | Status detail |
| --- | --- | --- | --- |
| CLI deployment | supported | supported | shared sync/RTC/infer-only loops |
| Web deployment | supported | supported | shared `RobotWebRuntime` profile path |
| CAMERA_PREVIEW | supported | supported | no Robot or PolicyClient |
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
open-loop metrics must use `Robot`, `ActionSpec`, state fields and camera roles.
The remaining Piper-specific reset/CAN/camera field wiring is isolated as a
known Web-profile migration item; it must not be copied into new shared code.
See `piper_rm2_generality_audit.md` for the exact search scope and remaining
compatibility wiring.

# Robot Capability Matrix

This matrix separates software support from hardware validation. A code path is
not treated as real-robot support until the matching hardware gate is recorded.

| Capability | Mock/Fake CI | Piper hardware | RM2 hardware | Current status |
| --- | --- | --- | --- | --- |
| CLI deployment | covered by help/import tests | not claimed by H6 | operator-validated RM2 CLI defaults | supported, deployment-specific |
| Web deployment | covered by fake Web runtime tests | not claimed by H6 | not claimed by H6 | supported software path |
| Camera preview | covered without Robot/PolicyClient | not claimed by H6 | not claimed by H6 | supported software path |
| Sync / RTC / infer-only runtime | covered by fake runtime tests | not claimed by H6 | RM2 CLI command path operator-validated | supported software path |
| Policy warmup and first live chunk prefetch | covered by Web/runtime tests | not claimed by H6 | not claimed by H6 | supported software path |
| LeRobot v2.1 recording | covered with FakeRobot/FakeCamera | not claimed by H6 | not claimed by H6 | requires `recording` extra |
| Offline data view | covered without hardware | not applicable | not applicable | requires data files and recording deps |
| Open-loop evaluation | covered without Robot | not applicable | not applicable | requires data/evaluation deps and policy server |
| Baseline | covered without hardware | not hardware motion | not hardware motion | supported metadata workflow |
| Move to recorded state | covered with FakeRobot/MockArm | hardware blocked | hardware blocked | experimental |
| Robot trajectory replay | covered with delayed FakeRobot | hardware blocked | hardware blocked | experimental |

## Hardening Gates

- H1 aligned recordings can be used for training only after validation confirms
  `mp_real.control_step_aligned=true` and no unexplained audit risk remains.
- H4/H5 real move-to-state and replay remain hardware blocked until the
  robot-specific safety profile, health, feedback freshness, stop capability
  and tracking gates are completed.
- H6 core deployments do not require `av` or `pyarrow`; data/recording tools
  require the `recording`, `data`, or `evaluation` optional extras.


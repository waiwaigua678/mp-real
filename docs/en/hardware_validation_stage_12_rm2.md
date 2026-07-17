# Stage 12 RM2 Baseline validation

This is independent of Piper validation and is not permission to move RM2.
Complete RM2-specific pose/replay safety gates, workspace clearance, joint-unit
review and stop verification before any motion-capable step.

1. Create a black-camera Baseline without starting an episode; verify RM2
   ActionSpec, camera roles, policy joint units and all runtime metadata.
2. Clone exactly one experimental change and review the Baseline diff.
3. Create a Baseline-backed evaluation and verify that it only enters manual
   preparation; no action is sent until the operator explicitly starts a round.
4. After approval, use 3–5 smoke trials, then 10 screening trials, then 20–30
   trials per configuration for comparison. Retain invalid, timeout and safety
   abort counts rather than suppressing them.

A MockArm pass is never hardware validation. Stop via RM2's verified physical
stop or emergency procedure on any unexpected movement or communication issue.

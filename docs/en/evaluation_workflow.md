# Evaluation workflow

The six workflows below are intentionally distinct:

- **Data view** reads recorded files only; no Robot, camera, or PolicyClient.
- **Move to recorded state** reads `observation.state`, preflights a pose plan,
  and requires separate hardware safety gates before movement.
- **Deployment from recorded state** hands a verified pose connection to fresh
  cameras and policy warmup; it never uses a recorded frame as a live action.
- **Robot trajectory replay** is policy-free replay of a reviewed action plan.
- **Open-loop evaluation** is teacher-forced policy inference over recorded
  observations and never executes an action.
- **Real-robot evaluation** uses `EvaluationSession`, manual labels and the
  shared runtime; a Baseline makes repeated rounds reproducible.

For each formal Baseline, record the selected ID, completed count, dataset path,
operator, exceptions and aborts. Start with 3–5 smoke trials, use 10 trials for
screening, at least 20–30 per configuration for comparison, and 50+ for a more
stable conclusion.

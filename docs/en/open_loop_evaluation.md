# Open-loop evaluation

`mp-open-loop-eval` is offline teacher-forced evaluation. It may contact a
policy server but never imports a robot SDK, creates a Robot, or sends an
action. Attaching its `summary.json` and `config.json` preserves dataset,
episode selection, target source, alignment, ActionSpec, state schema and
camera-role contract. Open-loop results with different contracts remain
separate and are never pooled in a Baseline A/B report.

Open-loop error is not real-robot success rate. Compare it alongside, never as
a substitute for, manually labelled real-robot Baseline results.

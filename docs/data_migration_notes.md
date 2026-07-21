# Data Migration Notes

## H1-Before Data

Datasets created before H1 may have used the policy inference observation for
multiple actions from the same action chunk. That can duplicate state and image
values across several control actions even though the robot continued moving.

Do not assume those rows are control-step aligned.

## Training Use

Legacy or unknown-semantics datasets are No-Go for direct training by default.
They may still be useful for:

- camera/debug inspection
- timing analysis
- policy-output telemetry review
- manual labels and Baseline references

They should not be used as supervised action rows unless an operator has
audited the alignment risk and accepted the limitation.

## Move-To-State And Replay

Recorded states may still be useful for visual inspection, but legacy metadata
does not make move-to-state or replay trustworthy. Command replay requires
known action source, action mode, ActionSpec, joint unit, arm count and gripper
layout. Unknown schemas remain rejected by planning or safety validation.

## Auditing

Run metadata validation:

```bash
uv run --extra recording mp-data-validate <dataset>
```

Run alignment risk detection:

```bash
uv run --extra recording mp-data-audit <dataset>
```

The audit reports repeated policy observation ids, repeated camera frame ids and
identical consecutive robot states. These are risk signals, not automatic proof
of corruption. A stationary robot or a camera backend that legitimately reuses
the latest frame can produce repeated values.

## Why Old Data Cannot Be Automatically Fixed

If the original per-control-step robot state or camera frame was not recorded,
it cannot be reconstructed from the policy observation or from the action chunk.
H6 therefore documents and detects legacy risk; it does not rewrite old data.


# Stage 11 validation: teacher-forced open-loop evaluation

## Safety boundary

`mp-open-loop-eval` and the optional job in `mp-data-view` are offline policy
evaluation paths. They do not create a `Robot`, camera, robot registry entry,
or robot-specific SDK object, and they never call an action execution method.
They may connect to the supplied policy WebSocket only after an operator starts
the command/job.

## Required real-data validation

Run each command with an available policy server; this is not a robot-motion
test.

1. Validate the target dataset first:

   ```bash
   uv run mp-data-validate recordings/<piper-dataset>
   uv run mp-data-validate recordings/<rm2-dataset>
   ```

2. Run one Piper episode and one RM2 episode for each of two policy labels.
   Keep every policy in its own output directory.

   ```bash
   uv run mp-open-loop-eval --dataset recordings/<piper-dataset> --episode 0 \
     --policy-url ws://<policy-host> --policy-label checkpoint-a \
     --target-source action --alignment sample_index \
     --output open_loop_results/piper-checkpoint-a

   uv run mp-open-loop-eval --dataset recordings/<rm2-dataset> --episode 0 \
     --policy-url ws://<policy-host> --policy-label checkpoint-b \
     --target-source action --alignment timestamp --max-timestamp-error 0.05 \
     --output open_loop_results/rm2-checkpoint-b
   ```

3. Inspect `config.json`, `summary.json`, every `reports/episode_*.json`, and
   the matching `predictions/*.npz`. Verify `teacher_forced=true`, one target
   source per result root, the expected ActionSpec/state schema, discarded
   warmup behavior, valid-mask tail handling, and source-dataset immutability.

4. Use absolute-control-step alignment only when `frame_index` is explicitly
   approved as the recorded control step and every evaluated source sample has
   non-negative `mp_real.chunk_cursor` telemetry. Pass
   `--allow-frame-index-as-control-step`; otherwise use sample-index or
   timestamp alignment.

## Interpretation limits

Open-loop error is not real-robot success rate. Teacher forcing always uses
the recorded state/images/prompt and therefore cannot expose every closed-loop
drift or error accumulation. A multimodal policy can choose an action that
differs from the recorded expert while still being useful. Review gripper event
timing and chunk-overlap consistency alongside pointwise MAE, and never merge
results from different ActionSpecs or target sources.

## Automated checks run for this implementation

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run mp-open-loop-eval --help
```

No real policy server or recorded Piper/RM2 deployment dataset was contacted
while writing this implementation. No real-hardware verification is required
for the Stage 11 offline evaluator itself.

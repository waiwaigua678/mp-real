# Stage 12 Piper Baseline validation

This checklist is not permission to move a robot. Obtain explicit approval for
each motion-capable command and complete Stage 9/10 gates first.

1. With black cameras and no action start, create a Baseline. Verify Git commit,
   policy label, checkpoint hash or `null`, ActionSpec, camera/RTC/warmup and
   initial-position fields.
2. Clone it with one parameter change. Verify the categorized diff and lineage.
3. From an unchanged Baseline, create an evaluation session. Confirm it remains
   in `PREPARING`; no warmup, reset, or action is issued.
4. After explicit motion approval, run 3–5 unloaded smoke trials. Confirm every
   terminal session attaches its compact result and dataset reference.
5. Repeat independently for RTC on/off or another single primary variable. Do
   not interpret 1/1 as a stable result. Use 20–30 trials per configuration for
   a formal comparison and 50+ for a stable conclusion.

Stop with the vendor physical stop or Web emergency controls for unexpected
motion, tracking error, health error, policy error or loss of communication.
Record the Baseline IDs, operator, aborts and dataset paths before resuming.

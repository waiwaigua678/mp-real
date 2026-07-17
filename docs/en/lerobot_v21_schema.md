# LeRobot v2.1 schema and Baselines

Recorded mp-real datasets preserve a versioned ActionSpec with field names,
units and semantics. Baselines retain that contract as a compact JSON snapshot;
they do not duplicate episode Parquet or video data. `ActionSpec` now has a
schema version and action mode while preserving legacy construction. Action and
state names, arm count and gripper indices are derived from `VectorField` and
round-trip through the recording schema.

Only identical declared contracts may be merged for A/B aggregation. Explicit
mapping remains required for a different state layout or unit.

# Robot replay safety

Trajectory replay is separate from policy deployment and from Baseline creation.
It requires an offline reviewed plan, an exclusive robot-control lease and a
second confirmation. The default Piper and RM2 adapters deliberately reject
physical execution until workspace, health, joint-limit and stop validation are
configured. See `hardware_validation_stage_10.md`.

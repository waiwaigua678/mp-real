# Known limitations

- Baseline resource arbitration is in-process; separate controller processes
  require deployment-level coordination outside this repository.
- Baseline comparison reports only metrics that were recorded. Tracking error,
  grasp success and placement success are shown as unavailable when no explicit
  telemetry or manual labels exist.
- Current Piper/RM2 pose and replay adapters are not hardware-validated and
  intentionally block movement without vendor-specific safety validation.
- CLI deployment paths do not yet use the Web policy startup coordinator.
- See `robot_capability_matrix.md` for the current Piper/RM2 software versus
  hardware-validation status.

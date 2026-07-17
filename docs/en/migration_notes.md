# Migration notes

Stage 12 adds `mp-baseline` and a Web Baseline page without changing existing
Piper/RM2 CLI names, the shared inference loops, policy client, or recording
schema location. Existing `ActionSpec(...)` calls remain valid because the new
metadata fields were appended with defaults.

The deprecated RM2-only Web implementation remains present for compatibility,
but normal `mp-real-web --robot rm2` uses the shared profile runtime.
`piper_rm2_generality_audit.md` records the remaining Piper-shaped Web wiring
that should be moved behind profile-owned interfaces in a future migration.

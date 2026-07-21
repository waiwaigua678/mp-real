# Recorder Memory Model After H2

H2 hardens the LeRobot v2.1 recorder without changing the standard Parquet row
semantics: `observation.state` and `action` still describe the executed
control step. The mp-real telemetry extension now streams bounded part files
instead of keeping whole-episode telemetry in memory.

H6 keeps Parquet and video dependencies outside the core deployment install.
Recording, validation, inspection, audit and offline data viewing require a
data extra such as `uv sync --extra recording`.

## Pre-H2 Growth Model

The old recorder worker kept several episode-scoped structures until recorder
shutdown:

- `observations`: full `ObservationCaptured` payloads, including images.
- `inference_latency_ns`: one entry per policy observation.
- `selected_actions` and `stabilized_actions`: incomplete event joins.
- `_EpisodeWriter` telemetry lists: camera timing, chunk ids, safety flags and
  raw chunks.
- `_EpisodeWriter._raw_chunks`: copied policy chunks, later duplicated into a
  dense `[N, H, D]` array at `close()`.

Measured with 640x480 RGB frames, 30 FPS, action dim 16, horizon 5:

| Cameras | Observation Cache | 10 Minutes | 60 Minutes | Queue 128 Events |
|---|---:|---:|---:|---:|
| 1 | ~0.880 MiB/frame | ~15.47 GiB | ~92.84 GiB | ~112.90 MiB |
| 3 | ~2.639 MiB/frame | ~46.38 GiB | ~278.29 GiB | ~337.98 MiB |

The old telemetry lists were smaller but still episode-length dependent:

| Cameras | Telemetry Lists 10 Minutes | Telemetry Lists 60 Minutes |
|---|---:|---:|
| 1 | ~12.5 MiB | ~74.7 MiB |
| 3 | ~13.4 MiB | ~80.5 MiB |

Raw chunk dense finalize added another temporary allocation. For RM2-like
16-dimensional actions, 60 minutes at 30 FPS is about 6.6 MiB of raw values,
plus Python object overhead from every chunk.

## H2 Ownership And Release Rules

Recorder-owned temporary caches are bounded by `RecorderConfig`:

- `max_observation_cache_entries`
- `max_inference_latency_entries`
- `max_incomplete_event_entries`

Each cache evicts oldest entries when full and is cleared at episode end. A
matched legacy observation is popped immediately after its action row is
recorded. H1 `ControlStepRecorded` rows do not require long-lived policy
observation payloads.

The event queue remains bounded by `queue_size`. `emit()` never blocks the
control thread. When full, the recorder records dropped event/frame counters;
`drop_policy="abort"` additionally marks the recorder failed and notifies the
evaluation service.

## Telemetry Parts

When `save_telemetry=true`, telemetry is written incrementally:

```text
telemetry/chunk-000/episode_000000/
  part_000000.npz
  part_000001.npz
  index.json
```

Each part contains a bounded number of control steps:

- control step ids
- policy observation/request/chunk ids
- camera frame ids, timestamps, reused flags and ages
- safety flags
- raw action chunks and raw chunk lengths

The writer pads raw chunks only within the current part. It no longer builds a
whole-episode dense `[N, H, D]` array at finalize.

`LeRobotV21EpisodeSource.get_episode_telemetry()` supports both H2 part layout
and legacy single-file `.npz`. `get_sample_telemetry()` reads only the relevant
part files for viewer sample inspection.

## Metrics

`LeRobotV21EpisodeRecorder.metrics()` reports:

- queue size, capacity and high watermark
- cache entry count, high watermark and eviction count
- buffered image bytes
- buffered telemetry bytes
- written frame count
- dropped frame/event count
- writer latency P50/P95
- current memory estimate
- telemetry part count
- active writer count

Evaluation status exposes these fields under `recording.metrics`.

## Validation Commands

Fast fake-unit coverage:

```bash
uv run python -m unittest tests.test_lerobot_v21 -v
```

Configurable local soak examples:

```bash
uv run python scripts/recorder_soak.py --duration-minutes 30 --cameras 1
uv run python scripts/recorder_soak.py --duration-minutes 60 --cameras 3
```

The soak script creates no robot and no policy client. It emits fake
`ControlStepRecorded` events with synthetic camera frames.

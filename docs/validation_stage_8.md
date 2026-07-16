# Stage 8 validation: offline data viewer

Date: 2026-07-16

The viewer was checked with the provided standard LeRobot v2.1 dataset:

```text
/home/pc4/.cache/huggingface/lerobot/local/piper_1armblowv01
```

Observed metadata:

- Robot: Piper
- Episodes: 165
- FPS: 30
- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`
- Episode 0: 344 samples
- This is standard LeRobot data without optional mp-real telemetry.

The local loopback HTTP smoke test verified catalog enumeration, episode
sample lookup, JPEG decoding of `cam_head`, and serving the Episode Viewer
page. It used no Robot, camera factory, policy client, CAN interface, or SDK.

Automated coverage in `tests/test_data_view.py` covers catalog access,
sample-index/timestamp/progress lookup, draggable-view cursor math,
play/pause rates, synchronized video/state metadata, absent videos, absent
telemetry, incomplete datasets, path traversal rejection, peak-preserving
downsampling, dynamic action dimensions, metrics, runtime-event display, and
selected-sample state.

Not covered by this data set:

- mp-real raw/stabilized/executed action and camera-age telemetry display;
- RM2 physical-camera recording;
- a truly long real-world single episode (the reader uses 512-row Parquet
  batches and does not cache decoded episode tables).

No real-hardware validation is required or performed for this stage.

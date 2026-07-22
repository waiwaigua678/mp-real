#!/usr/bin/env python3
"""Re-encode LeRobot videos with short GOPs and pad short episodes.

The script is intended for datasets where LeRobot random frame access is slow
because videos were encoded with long GOPs, and for the common off-by-one video
frame mismatch where the parquet episode length is one frame longer than the
video stream. It reads the expected frame counts from meta/episodes.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    chunk: int


@dataclass(frozen=True)
class VideoTask:
    episode: Episode
    video_key: str
    src: Path
    dst: Path
    expected_frames: int
    expected_width: int | None
    expected_height: int | None


@dataclass(frozen=True)
class ProbeResult:
    task: VideoTask
    exists: bool
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    error: str | None = None

    @property
    def frame_delta(self) -> int | None:
        if self.frame_count is None:
            return None
        return self.task.expected_frames - self.frame_count


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_rate(raw: str | None) -> float | None:
    if not raw or raw == "0/0":
        return None
    if "/" not in raw:
        try:
            return float(raw)
        except ValueError:
            return None
    numerator_raw, denominator_raw = raw.split("/", 1)
    try:
        numerator = float(numerator_raw)
        denominator = float(denominator_raw)
    except ValueError:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def parse_optional_int(raw: Any) -> int | None:
    if raw in (None, "N/A"):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"required command not found: {name}")


def read_episodes(root: Path, chunks_size: int) -> list[Episode]:
    rows = read_jsonl(root / "meta/episodes.jsonl")
    episodes: list[Episode] = []
    for row in rows:
        episode_index = int(row["episode_index"])
        episodes.append(
            Episode(
                index=episode_index,
                length=int(row["length"]),
                chunk=episode_index // chunks_size,
            )
        )
    return sorted(episodes, key=lambda item: item.index)


def discover_video_keys(info: dict[str, Any], requested: list[str] | None) -> list[str]:
    features = info.get("features", {})
    video_keys = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    video_keys.sort()
    if requested:
        missing = sorted(set(requested) - set(video_keys))
        if missing:
            raise ValueError(f"requested video keys are not in info.json: {missing}")
        return requested
    return video_keys


def feature_size(info: dict[str, Any], video_key: str) -> tuple[int | None, int | None]:
    feature = info.get("features", {}).get(video_key, {})
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return None, None


def video_relpath(template: str, episode: Episode, video_key: str) -> Path:
    return Path(
        template.format(
            episode_chunk=episode.chunk,
            episode_index=episode.index,
            video_key=video_key,
        )
    )


def build_tasks(
    root: Path,
    temp_root: Path,
    info: dict[str, Any],
    episodes: list[Episode],
    video_keys: list[str],
) -> list[VideoTask]:
    template = str(info["video_path"])
    tasks: list[VideoTask] = []
    for episode in episodes:
        for video_key in video_keys:
            relpath = video_relpath(template, episode, video_key)
            width, height = feature_size(info, video_key)
            tasks.append(
                VideoTask(
                    episode=episode,
                    video_key=video_key,
                    src=root / relpath,
                    dst=temp_root / relpath.relative_to("videos"),
                    expected_frames=episode.length,
                    expected_width=width,
                    expected_height=height,
                )
            )
    return tasks


def probe_video(task: VideoTask, *, use_dst: bool = False) -> ProbeResult:
    path = task.dst if use_dst else task.src
    if not path.exists():
        return ProbeResult(task=task, exists=False, error=f"missing file: {path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        return ProbeResult(task=task, exists=True, error=proc.stderr.strip())
    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return ProbeResult(task=task, exists=True, error=f"invalid ffprobe output: {exc}")

    frame_count = parse_optional_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = parse_optional_int(stream.get("nb_frames"))
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    return ProbeResult(
        task=task,
        exists=True,
        frame_count=frame_count,
        fps=fps,
        width=parse_optional_int(stream.get("width")),
        height=parse_optional_int(stream.get("height")),
        codec=stream.get("codec_name"),
    )


def probe_keyframes(path: Path) -> tuple[int, float | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time",
        "-of",
        "json",
        str(path),
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    payload = json.loads(proc.stdout)
    times: list[float] = []
    for frame in payload.get("frames", []):
        raw = frame.get("best_effort_timestamp_time") or frame.get("pkt_pts_time")
        if raw not in (None, "N/A"):
            times.append(float(raw))
    times.sort()
    if len(times) < 2:
        return len(times), None
    max_gap = max(current - previous for previous, current in zip(times, times[1:]))
    return len(times), max_gap


def parallel_probe(tasks: list[VideoTask], jobs: int, *, use_dst: bool = False) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(probe_video, task, use_dst=use_dst) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done == len(futures) or done % 25 == 0:
                print(f"probed {done}/{len(futures)} videos")
    return sorted(results, key=lambda item: (item.task.episode.index, item.task.video_key))


def is_fps_ok(actual: float | None, expected: float, tolerance: float = 0.01) -> bool:
    return actual is not None and abs(actual - expected) <= tolerance


def summarize_probes(results: list[ProbeResult], expected_fps: float) -> dict[str, list[ProbeResult]]:
    buckets = {
        "missing": [],
        "probe_error": [],
        "frame_ok": [],
        "short": [],
        "long": [],
        "unknown_frames": [],
        "fps_mismatch": [],
        "size_mismatch": [],
    }
    for result in results:
        if not result.exists:
            buckets["missing"].append(result)
            continue
        if result.error:
            buckets["probe_error"].append(result)
            continue
        if result.frame_count is None:
            buckets["unknown_frames"].append(result)
        elif result.frame_count < result.task.expected_frames:
            buckets["short"].append(result)
        elif result.frame_count > result.task.expected_frames:
            buckets["long"].append(result)
        else:
            buckets["frame_ok"].append(result)

        if not is_fps_ok(result.fps, expected_fps):
            buckets["fps_mismatch"].append(result)
        if (
            result.task.expected_width is not None
            and result.width is not None
            and result.width != result.task.expected_width
        ) or (
            result.task.expected_height is not None
            and result.height is not None
            and result.height != result.task.expected_height
        ):
            buckets["size_mismatch"].append(result)
    return buckets


def print_summary(title: str, buckets: dict[str, list[ProbeResult]]) -> None:
    print(title)
    for name in (
        "frame_ok",
        "short",
        "long",
        "missing",
        "probe_error",
        "unknown_frames",
        "fps_mismatch",
        "size_mismatch",
    ):
        print(f"  {name}={len(buckets[name])}")


def print_examples(label: str, results: list[ProbeResult], limit: int = 20) -> None:
    if not results:
        return
    print(f"{label}:")
    for result in results[:limit]:
        task = result.task
        print(
            "  "
            f"episode={task.episode.index:06d} key={task.video_key} "
            f"frames={result.frame_count} expected={task.expected_frames} "
            f"fps={result.fps} codec={result.codec} path={task.src}"
        )
    if len(results) > limit:
        print(f"  ... {len(results) - limit} more")


def validate_source_results(
    buckets: dict[str, list[ProbeResult]],
    *,
    max_pad_frames: int,
    allow_trim: bool,
) -> None:
    fatal: list[str] = []
    if buckets["missing"]:
        fatal.append(f"missing videos: {len(buckets['missing'])}")
    if buckets["probe_error"]:
        fatal.append(f"ffprobe errors: {len(buckets['probe_error'])}")
    if buckets["unknown_frames"]:
        fatal.append(f"unknown frame counts: {len(buckets['unknown_frames'])}")
    large_short = [
        result
        for result in buckets["short"]
        if result.frame_delta is not None and result.frame_delta > max_pad_frames
    ]
    if large_short:
        fatal.append(f"videos short by more than --max-pad-frames: {len(large_short)}")
    if buckets["long"] and not allow_trim:
        fatal.append(f"videos longer than metadata length: {len(buckets['long'])}")
    if fatal:
        raise RuntimeError("; ".join(fatal))


def reencode_video(
    result: ProbeResult,
    *,
    fps: float,
    gop: int,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
    allow_trim: bool,
) -> None:
    task = result.task
    if result.frame_count is None:
        raise RuntimeError(f"cannot re-encode without source frame count: {task.src}")

    filters = [f"fps=fps={fps:g}"]
    if result.frame_count < task.expected_frames:
        missing = task.expected_frames - result.frame_count
        pad_duration = (missing + 1) / fps
        filters.append(f"tpad=stop_mode=clone:stop_duration={pad_duration:.9f}")
    elif result.frame_count > task.expected_frames and not allow_trim:
        raise RuntimeError(f"refusing to trim longer video: {task.src}")

    task.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.dst.with_name(f"{task.dst.stem}.inprogress{task.dst.suffix}")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(task.src),
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(task.expected_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-bf",
        "0",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "-threads",
        str(ffmpeg_threads),
        str(tmp),
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"ffmpeg failed for {task.src}: {proc.stderr.strip()}")
    tmp.replace(task.dst)


def parallel_reencode(
    results: list[ProbeResult],
    jobs: int,
    *,
    fps: float,
    gop: int,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
    allow_trim: bool,
) -> None:
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                reencode_video,
                result,
                fps=fps,
                gop=gop,
                crf=crf,
                preset=preset,
                ffmpeg_threads=ffmpeg_threads,
                allow_trim=allow_trim,
            )
            for result in results
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done == len(futures) or done % 10 == 0:
                print(f"encoded {done}/{len(futures)} videos")


def validate_outputs(results: list[ProbeResult], expected_fps: float) -> None:
    buckets = summarize_probes(results, expected_fps)
    print_summary("output_summary", buckets)
    bad = (
        buckets["missing"]
        + buckets["probe_error"]
        + buckets["unknown_frames"]
        + buckets["short"]
        + buckets["long"]
        + buckets["fps_mismatch"]
        + buckets["size_mismatch"]
    )
    if bad:
        print_examples("output_validation_issues", bad)
        raise RuntimeError("encoded videos did not pass validation; leaving temp directory in place")


def swap_video_dirs(root: Path, temp_root: Path, stamp: str) -> Path:
    videos_root = root / "videos"
    backup_root = root / f"videos.backup_before_reencode_{stamp}"
    if backup_root.exists():
        raise FileExistsError(f"backup already exists: {backup_root}")
    videos_root.rename(backup_root)
    try:
        temp_root.rename(videos_root)
    except Exception:
        backup_root.rename(videos_root)
        raise
    return backup_root


def inspect_keyframe_samples(tasks: list[VideoTask], *, use_dst: bool, limit: int) -> None:
    if limit <= 0:
        return
    print("keyframe_samples:")
    for task in tasks[:limit]:
        path = task.dst if use_dst else task.src
        count, max_gap = probe_keyframes(path)
        print(
            "  "
            f"episode={task.episode.index:06d} key={task.video_key} "
            f"keyframes={count} max_gap_sec={max_gap}"
        )


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset root, e.g. ~/.cache/huggingface/lerobot/local/rm2_stack_cupv02.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually re-encode and swap videos. Default only probes and reports.",
    )
    parser.add_argument(
        "--video-key",
        action="append",
        default=None,
        help="Video key to process. Can be passed multiple times. Default: all video features.",
    )
    parser.add_argument("--gop", type=positive_int, default=2, help="Output GOP/keyframe interval. Default: 2.")
    parser.add_argument("--crf", type=int, default=23, help="libx264 CRF. Default: 23.")
    parser.add_argument("--preset", default="veryfast", help="libx264 preset. Default: veryfast.")
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=max(1, min(4, (os.cpu_count() or 4) // 4)),
        help="Parallel ffprobe/ffmpeg jobs. Default: min(4, cpu_count//4).",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Threads per ffmpeg process. 0 lets ffmpeg choose automatically. Default: 0.",
    )
    parser.add_argument(
        "--max-pad-frames",
        type=int,
        default=5,
        help="Abort if a source video is short by more than this many frames. Default: 5.",
    )
    parser.add_argument(
        "--allow-trim",
        action="store_true",
        help="Allow videos longer than metadata to be trimmed to the episode length.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Probe only the first N videos. For dry-run/debug only; not allowed with --apply.",
    )
    parser.add_argument(
        "--inspect-keyframes",
        type=int,
        default=3,
        help="Inspect keyframe count/max gap for the first N videos. Default: 3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply and args.limit is not None:
        raise ValueError("--limit is only allowed for dry-run/debug")
    if args.max_pad_frames < 0:
        raise ValueError("--max-pad-frames must be >= 0")

    require_tool("ffmpeg")
    require_tool("ffprobe")

    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    videos_root = root / "videos"
    if not info_path.exists():
        raise FileNotFoundError(f"missing info.json: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"missing episodes.jsonl: {episodes_path}")
    if not videos_root.exists():
        raise FileNotFoundError(f"missing videos directory: {videos_root}")

    info = read_json(info_path)
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = read_episodes(root, chunks_size)
    total_episodes = int(info["total_episodes"])
    if len(episodes) != total_episodes:
        raise ValueError(f"episodes.jsonl has {len(episodes)} rows, info.json says {total_episodes}")

    video_keys = discover_video_keys(info, args.video_key)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_root = root / f"videos.reencoded_tmp_{stamp}"
    tasks = build_tasks(root, temp_root, info, episodes, video_keys)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"dataset_root={root}")
    print(f"episodes={len(episodes)}")
    print(f"total_frames={sum(episode.length for episode in episodes)}")
    print(f"video_keys={video_keys}")
    print(f"videos_to_process={len(tasks)}")
    print(f"fps={fps:g}")
    print(f"gop={args.gop}")
    print(f"jobs={args.jobs}")

    source_results = parallel_probe(tasks, args.jobs, use_dst=False)
    source_buckets = summarize_probes(source_results, fps)
    print_summary("source_summary", source_buckets)
    print_examples("short_videos", source_buckets["short"])
    print_examples("long_videos", source_buckets["long"])
    print_examples("source_probe_issues", source_buckets["missing"] + source_buckets["probe_error"])
    inspect_keyframe_samples(tasks, use_dst=False, limit=args.inspect_keyframes)

    if not args.apply:
        print("dry_run=true; add --apply to re-encode and swap videos")
        return 0

    validate_source_results(
        source_buckets,
        max_pad_frames=args.max_pad_frames,
        allow_trim=args.allow_trim,
    )
    if temp_root.exists():
        raise FileExistsError(f"temp directory already exists: {temp_root}")

    print(f"temp_videos={temp_root}")
    parallel_reencode(
        source_results,
        args.jobs,
        fps=fps,
        gop=args.gop,
        crf=args.crf,
        preset=args.preset,
        ffmpeg_threads=args.ffmpeg_threads,
        allow_trim=args.allow_trim,
    )

    output_results = parallel_probe(tasks, args.jobs, use_dst=True)
    validate_outputs(output_results, fps)
    inspect_keyframe_samples(tasks, use_dst=True, limit=args.inspect_keyframes)
    backup_root = swap_video_dirs(root, temp_root, stamp)
    print(f"backup_videos={backup_root}")
    print(f"updated_videos={videos_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

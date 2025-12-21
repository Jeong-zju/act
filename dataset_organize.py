#!/usr/bin/env python3
"""
Convert a LeRobot dataset into ACT's HDF5 episodic format, optionally splitting
episodes into fixed-length parts and rebasing timestamps so each part starts at
timestamp=0.0.

Supports:
  - LeRobot v2.1 (per-episode parquet + per-episode mp4 videos)
  - LeRobot v3.0 (single parquet file(s) containing all episodes; videos are
    concatenated per file, indexed by the global `index` column)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _require_parquet_reader():
    try:
        import pyarrow.parquet as pq  # type: ignore

        return ("pyarrow", pq)
    except Exception:
        try:
            import pandas as pd  # type: ignore

            return ("pandas", pd)
        except Exception as exc:
            raise RuntimeError(
                "Reading LeRobot parquet files requires either 'pyarrow' or 'pandas'.\n"
                "Install one of:\n"
                "  pip install pyarrow\n"
                "  pip install pandas pyarrow"
            ) from exc


def _stack_object_array(values: np.ndarray) -> np.ndarray:
    if values.dtype != object:
        return values
    return np.stack([np.asarray(v) for v in values], axis=0)


def _video_backend_order(backend: str, *, codec: Optional[str]) -> List[str]:
    if backend != "auto":
        return [backend]
    if codec is not None and codec.lower() == "av1":
        # OpenCV often cannot decode AV1 and emits noisy ffmpeg logs; prefer other backends.
        return ["pyav", "imageio"]
    return ["opencv", "pyav", "imageio"]


def _read_video_frame(video_path: Path, frame_index: int, *, backend: str = "auto", codec: Optional[str] = None) -> np.ndarray:
    backend_order = _video_backend_order(backend, codec=codec)

    last_err: Optional[BaseException] = None
    for candidate in backend_order:
        try:
            if candidate == "opencv":
                import cv2  # type: ignore

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError("cv2.VideoCapture failed to open")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                cap.release()
                if not ok or frame_bgr is None:
                    raise RuntimeError("cv2 failed to decode requested frame (likely unsupported codec)")
                return frame_bgr[:, :, ::-1].astype(np.uint8, copy=False)

            if candidate == "pyav":
                import av  # type: ignore

                container = av.open(str(video_path))
                try:
                    for i, frame in enumerate(container.decode(video=0)):
                        if i == int(frame_index):
                            return frame.to_ndarray(format="rgb24")
                finally:
                    container.close()
                raise RuntimeError("PyAV decoded video but frame index was out of range")

            if candidate == "imageio":
                import imageio.v3 as iio  # type: ignore

                frame = iio.imread(str(video_path), index=int(frame_index))
                if frame.ndim == 3 and frame.shape[-1] >= 3:
                    return frame[:, :, :3].astype(np.uint8, copy=False)
                raise RuntimeError("imageio returned unexpected frame shape")

            raise ValueError(f"Unknown video backend: {candidate}")
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(
        "Failed to decode LeRobot video frame.\n"
        "If your dataset uses AV1 mp4 (common for LeRobot), OpenCV often can't decode it.\n"
        "Try one of:\n"
        "  pip install av\n"
        "  pip install imageio imageio-ffmpeg\n"
        "Or re-encode the videos to H.264 with ffmpeg, or run with --skip_images."
    ) from last_err


def _compute_qvel(qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros_like(qpos, dtype=np.float32)
    if len(qpos) >= 2:
        qvel[1:] = (qpos[1:] - qpos[:-1]) * float(fps)
    return qvel


def _split_indices(total_len: int, segment_len: Optional[int]) -> List[Tuple[int, int]]:
    if segment_len is None or segment_len <= 0 or segment_len >= total_len:
        return [(0, total_len)]
    out: List[Tuple[int, int]] = []
    start = 0
    while start < total_len:
        end = min(total_len, start + int(segment_len))
        out.append((start, end))
        start = end
    return out


def _default_output_dir(input_dir: Path) -> Path:
    # Save under repo-local ./dataset by default.
    repo_root = Path(__file__).resolve().parent
    return repo_root / "dataset" / f"{input_dir.name}_act"


@dataclass(frozen=True)
class EpisodeData:
    action: np.ndarray  # (T, A)
    state: np.ndarray  # (T, S)
    timestamp: np.ndarray  # (T,)
    frame_index: np.ndarray  # (T,)


def _load_lerobot_v21_episode(dataset_root: Path, info: Dict, episode_index: int) -> EpisodeData:
    backend, reader = _require_parquet_reader()
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    parquet_rel = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
    parquet_path = dataset_root / parquet_rel

    if backend == "pyarrow":
        table = reader.read_table(parquet_path)
        action_col = table.column("action").to_pylist()
        state_col = table.column("observation.state").to_pylist()
        ts_col = table.column("timestamp").to_pylist()
        frame_col = table.column("frame_index").to_pylist() if "frame_index" in table.column_names else None
    else:
        df = reader.read_parquet(parquet_path)
        action_col = df["action"].to_list()
        state_col = df["observation.state"].to_list()
        ts_col = df["timestamp"].to_list()
        frame_col = df["frame_index"].to_list() if "frame_index" in df.columns else None

    action = np.asarray(action_col, dtype=np.float32)
    state = np.asarray(state_col, dtype=np.float32)
    timestamp = np.asarray(ts_col, dtype=np.float32)
    if timestamp.ndim == 2 and timestamp.shape[1] == 1:
        timestamp = timestamp[:, 0]
    if action.ndim == 1:
        action = _stack_object_array(action)
    if state.ndim == 1:
        state = _stack_object_array(state)
    if frame_col is None:
        frame_index = np.arange(len(action), dtype=np.int64)
    else:
        frame_index = np.asarray(frame_col, dtype=np.int64)
        if frame_index.ndim == 2 and frame_index.shape[1] == 1:
            frame_index = frame_index[:, 0]

    return EpisodeData(action=action, state=state, timestamp=timestamp, frame_index=frame_index)


def _iter_lerobot_v21_episode_indices(dataset_root: Path) -> List[int]:
    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"Missing LeRobot v2.1 episodes file: {episodes_jsonl}")
    ids: List[int] = []
    with episodes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ids.append(int(item["episode_index"]))
    return ids


_V30_PARQUET_RE = re.compile(r"chunk-(?P<chunk>\d{3})/file-(?P<file>\d{3})\.parquet$")


def _parse_v30_chunk_file(parquet_path: Path) -> Tuple[int, int]:
    match = _V30_PARQUET_RE.search(str(parquet_path).replace("\\\\", "/"))
    if not match:
        raise ValueError(f"Unexpected v3.0 parquet path (can't parse chunk/file): {parquet_path}")
    return int(match.group("chunk")), int(match.group("file"))


def _load_lerobot_v30_rows(dataset_root: Path, parquet_path: Path) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
    backend, reader = _require_parquet_reader()
    if backend == "pyarrow":
        table = reader.read_table(parquet_path)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
    else:
        df = reader.read_parquet(parquet_path)
        cols = {name: df[name].to_list() for name in df.columns}

    out: Dict[str, np.ndarray] = {}
    for name, values in cols.items():
        arr = np.asarray(values)
        if arr.dtype == object:
            arr = _stack_object_array(arr)
        out[name] = arr

    chunk_file = _parse_v30_chunk_file(parquet_path.relative_to(dataset_root / "data"))
    return chunk_file, out


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_act_hdf5_episode(
    *,
    out_path: Path,
    sim: bool,
    camera_names: Sequence[str],
    episode_len: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    timestamp: Optional[np.ndarray],
    images_by_camera: Optional[Dict[str, np.ndarray]],
) -> None:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency: h5py (required to write ACT .hdf5 episodes).") from exc

    _ensure_parent_dir(out_path)

    with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = bool(sim)
        obs = root.create_group("observations")
        image_group = obs.create_group("images")

        if images_by_camera is not None:
            for cam_name in camera_names:
                arr = images_by_camera[cam_name]
                h, w = int(arr.shape[1]), int(arr.shape[2])
                _ = image_group.create_dataset(
                    cam_name,
                    (episode_len, h, w, 3),
                    dtype="uint8",
                    chunks=(1, h, w, 3),
                )
                image_group[cam_name][...] = arr

        _ = obs.create_dataset("qpos", (episode_len, qpos.shape[-1]), dtype="float32")
        _ = obs.create_dataset("qvel", (episode_len, qvel.shape[-1]), dtype="float32")
        _ = root.create_dataset("action", (episode_len, action.shape[-1]), dtype="float32")

        obs["qpos"][...] = qpos
        obs["qvel"][...] = qvel
        root["action"][...] = action

        if timestamp is not None:
            _ = obs.create_dataset("timestamp", (episode_len,), dtype="float32")
            obs["timestamp"][...] = timestamp


def _pad_to_len(x: np.ndarray, episode_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) >= episode_len:
        return x[:episode_len]
    pad_shape = (episode_len - len(x),) + x.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _pad_images_to_len(images: np.ndarray, episode_len: int) -> np.ndarray:
    if len(images) >= episode_len:
        return images[:episode_len]
    pad_shape = (episode_len - len(images),) + images.shape[1:]
    pad = np.zeros(pad_shape, dtype=images.dtype)
    return np.concatenate([images, pad], axis=0)


def _convert_episode_to_act_hdf5(
    *,
    out_dir: Path,
    out_episode_start: int,
    episode: EpisodeData,
    fps: float,
    camera_names: Sequence[str],
    video_paths: Optional[Dict[str, Path]],
    video_frame_indices: Optional[np.ndarray],
    segment_len: Optional[int],
    episode_len: int,
    video_backend: str,
    video_codec: Optional[str],
    skip_images: bool,
) -> int:
    segments = _split_indices(len(episode.action), segment_len)
    written = 0

    for seg_idx, (start, end) in enumerate(segments):
        part_action = episode.action[start:end]
        part_state = episode.state[start:end]
        part_ts = episode.timestamp[start:end].astype(np.float32, copy=False)
        if len(part_ts) > 0:
            part_ts = part_ts - float(part_ts[0])

        qpos = _pad_to_len(part_state.astype(np.float32, copy=False), episode_len)
        qvel = _pad_to_len(_compute_qvel(part_state.astype(np.float32, copy=False), fps), episode_len)
        action = _pad_to_len(part_action.astype(np.float32, copy=False), episode_len)
        timestamp = _pad_to_len(part_ts.astype(np.float32, copy=False), episode_len)

        images_by_camera: Optional[Dict[str, np.ndarray]] = None
        if not skip_images:
            if video_paths is None or video_frame_indices is None:
                raise ValueError("Images requested but video paths/indices were not provided.")
            images_by_camera = {}
            for cam_name in camera_names:
                video_path = video_paths[cam_name]
                frames = []
                for frame_idx in video_frame_indices[start:end]:
                    frames.append(
                        _read_video_frame(
                            video_path,
                            int(frame_idx),
                            backend=video_backend,
                            codec=video_codec,
                        )
                    )
                img_arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)
                images_by_camera[cam_name] = _pad_images_to_len(img_arr, episode_len)

        part_out = out_dir / f"episode_{out_episode_start + seg_idx:06d}.hdf5"

        _write_act_hdf5_episode(
            out_path=part_out,
            sim=False,
            camera_names=camera_names,
            episode_len=episode_len,
            qpos=qpos,
            qvel=qvel,
            action=action,
            timestamp=timestamp,
            images_by_camera=images_by_camera,
        )
        written += 1

        print(f"Write {part_out}.")

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the downloaded LeRobot dataset directory (contains meta/info.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for ACT hdf5 episodes (default: ./dataset/<input_name>_act).",
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["main", "secondary_0", "secondary_1"],
        help="Camera names; for LeRobot these map to observation.images.<name> video keys.",
    )
    parser.add_argument(
        "--segment_len",
        type=int,
        default=None,
        help="If set, split each episode into fixed-length parts.",
    )
    parser.add_argument(
        "--episode_len",
        type=int,
        default=None,
        help="Fixed length to store in each output hdf5 (defaults to segment_len, else per-episode length).",
    )
    parser.add_argument(
        "--video_backend",
        default="auto",
        choices=("auto", "opencv", "pyav", "imageio"),
        help="Backend used for decoding mp4 frames (AV1).",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip decoding and saving images (saves qpos/qvel/action only).",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes to convert.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()
    info_path = input_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing meta/info.json: {info_path}")

    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", "")).lower()
    fps = float(info.get("fps", 10))
    video_codec: Optional[str] = None
    try:
        features = info.get("features", {})
        # Prefer main camera feature if present.
        for key in ("observation.images.main", "observation.images.secondary_0", "observation.images.secondary_1"):
            if key in features:
                video_codec = features[key].get("info", {}).get("video.codec")
                break
    except Exception:
        video_codec = None

    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_names = list(args.camera_names)
    segment_len = args.segment_len

    total_written = 0
    next_out_episode = 0

    if version in ("v2.1", "2.1"):
        episode_indices = _iter_lerobot_v21_episode_indices(input_dir)
        if args.max_episodes is not None:
            episode_indices = episode_indices[: int(args.max_episodes)]

        for episode_index in episode_indices:
            episode = _load_lerobot_v21_episode(input_dir, info, episode_index)

            episode_len = args.episode_len
            if episode_len is None:
                episode_len = int(segment_len) if segment_len is not None else int(len(episode.action))

            chunks_size = int(info.get("chunks_size", 1000))
            episode_chunk = int(episode_index) // chunks_size
            video_paths: Dict[str, Path] = {}
            for cam in camera_names:
                video_key = cam if cam.startswith("observation.images.") else f"observation.images.{cam}"
                video_rel = info["video_path"].format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                video_paths[cam] = input_dir / video_rel

            written = _convert_episode_to_act_hdf5(
                out_dir=output_dir,
                out_episode_start=next_out_episode,
                episode=episode,
                fps=fps,
                camera_names=camera_names,
                video_paths=video_paths,
                video_frame_indices=episode.frame_index,
                segment_len=segment_len,
                episode_len=int(episode_len),
                video_backend=args.video_backend,
                video_codec=video_codec,
                skip_images=bool(args.skip_images),
            )
            total_written += written
            next_out_episode += written

    elif version in ("v3.0", "3.0"):
        data_root = input_dir / "data"
        parquet_files = sorted(data_root.glob("chunk-*/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under: {data_root}")

        for parquet_path in parquet_files:
            (chunk_index, file_index), cols = _load_lerobot_v30_rows(input_dir, parquet_path)
            if "episode_index" not in cols:
                raise KeyError(f"Expected 'episode_index' column in {parquet_path}")

            episode_index_arr = cols["episode_index"].astype(np.int64)
            frame_index_arr = cols.get("frame_index", None)
            if frame_index_arr is None:
                frame_index_arr = np.arange(len(episode_index_arr), dtype=np.int64)
            else:
                frame_index_arr = frame_index_arr.astype(np.int64).reshape(-1)

            timestamp_arr = cols.get("timestamp", None)
            if timestamp_arr is None:
                timestamp_arr = (frame_index_arr / float(fps)).astype(np.float32)
            else:
                timestamp_arr = timestamp_arr.astype(np.float32)
                if timestamp_arr.ndim == 2 and timestamp_arr.shape[1] == 1:
                    timestamp_arr = timestamp_arr[:, 0]

            action_arr = cols["action"].astype(np.float32)
            state_arr = cols["observation.state"].astype(np.float32)
            if action_arr.ndim == 1:
                action_arr = _stack_object_array(action_arr)
            if state_arr.ndim == 1:
                state_arr = _stack_object_array(state_arr)

            global_index_arr = cols.get("index", None)
            if global_index_arr is None:
                global_index_arr = frame_index_arr
            global_index_arr = global_index_arr.astype(np.int64).reshape(-1)

            unique_episodes = np.unique(episode_index_arr)
            if args.max_episodes is not None:
                unique_episodes = unique_episodes[: int(args.max_episodes)]

            # Video files are concatenated per (chunk_index, file_index)
            video_paths: Dict[str, Path] = {}
            for cam in camera_names:
                video_key = cam if cam.startswith("observation.images.") else f"observation.images.{cam}"
                video_rel = info["video_path"].format(
                    chunk_index=chunk_index,
                    file_index=file_index,
                    video_key=video_key,
                )
                video_paths[cam] = input_dir / video_rel

            for episode_index in unique_episodes:
                mask = episode_index_arr == int(episode_index)
                order = np.argsort(frame_index_arr[mask])

                episode = EpisodeData(
                    action=action_arr[mask][order],
                    state=state_arr[mask][order],
                    timestamp=timestamp_arr[mask][order],
                    frame_index=global_index_arr[mask][order],
                )

                episode_len = args.episode_len
                if episode_len is None:
                    episode_len = int(segment_len) if segment_len is not None else int(len(episode.action))

                written = _convert_episode_to_act_hdf5(
                    out_dir=output_dir,
                    out_episode_start=next_out_episode,
                    episode=episode,
                    fps=fps,
                    camera_names=camera_names,
                    video_paths=video_paths,
                    video_frame_indices=episode.frame_index,
                    segment_len=segment_len,
                    episode_len=int(episode_len),
                    video_backend=args.video_backend,
                    video_codec=video_codec,
                    skip_images=bool(args.skip_images),
                )
                total_written += written
                next_out_episode += written

    else:
        raise ValueError(f"Unsupported LeRobot codebase_version: {info.get('codebase_version')}")

    print(f"Saved {total_written} episode file(s) to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

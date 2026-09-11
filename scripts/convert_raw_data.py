"""Convert a LeRobot v2.x bi-Piper dataset (raw_data/<task>) into the X-WAM training format.

Works for any task dataset that shares the bi-Piper collection layout. Per episode this script:
  1. Transcodes RGB videos  : videos/chunk-000/<camera_key>/episode_XXXXXX.mp4 (AV1)
                              -> video/<view>/chunk-0000/episode_XXXXXXX.mp4 (H.264)
  2. Transcodes depth videos: depths/<camera>/episode-XXXXXX.mkv (FFV1 gray16le, uint16, scale_m)
                              -> depth/<view>/chunk-0000/episode_XXXXXXX.mp4 (H.264 lossless, 8-bit meters)
  3. Runs forward kinematics (scripts/piperx_fk.py) on joint states/actions to produce
     end-effector proprio/actions in the episode JSON.
  4. Writes metadata.json at the dataset root.

Output layout (matches configs/data/*.yaml expectations):
    <dst>/
    ├── metadata.json
    ├── data/chunk-0000/episode_XXXXXXX.json
    ├── video/<view>/chunk-0000/episode_XXXXXXX.mp4
    └── depth/<view>/chunk-0000/episode_XXXXXXX.mp4

Usage::

    python scripts/convert_raw_data.py \
        --src raw_data/<task>_v1 \
        --dst sft_datasets/<task> \
        [--workers 16] [--depth-max-m 3.0] [--depth-scale-m 0.001] [--use-gpu]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piperx_fk import PiperXForwardKinematics  # noqa: E402

CAMERA_VIEW_MAP = {
    "observation.images.fixed_front": ("fixed_front", "static"),
    "observation.images.left_arm": ("left_arm", "dynamic"),
    "observation.images.right_arm": ("right_arm", "dynamic"),
}
NUM_JOINTS = 6
# state/action layout: [left_j1..j6, left_gripper, right_j1..j6, right_gripper]

USE_GPU = False  # NVENC H.264 encoding for RGB/depth; requires ffmpeg with h264_nvenc


def run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:8])}...):\n{proc.stderr.decode()[-2000:]}")


def transcode_rgb(src: Path, dst: Path, fps: float) -> None:
    if USE_GPU:
        codec = ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "14", "-preset", "p4"]
    else:
        codec = ["-c:v", "libx264", "-crf", "14", "-preset", "medium"]
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", "format=yuv420p",
        *codec,
        "-r", f"{fps}", "-an", str(dst),
    ]
    run_ffmpeg(cmd)


def read_depth_u16(src: Path, width: int, height: int) -> np.ndarray:
    """Decode an FFV1 gray16le video into [T, H, W] uint16 via ffmpeg rawvideo pipe."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(src),
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg depth decode failed for {src}:\n{proc.stderr.decode()[-2000:]}")
    raw = proc.stdout
    num_frames = len(raw) // (width * height * 2)
    if num_frames == 0:
        raise RuntimeError(f"No depth frames decoded from {src}")
    return np.frombuffer(raw[: num_frames * width * height * 2], dtype="<u2").reshape(num_frames, height, width)


def encode_depth(depth_u16: np.ndarray, dst: Path, fps: float, scale_m: float, max_m: float) -> None:
    """Linearly map depth (meters) to 8-bit [0, 255] and encode lossless H.264.

    Meters outside [0, max_m] are clipped (65535-style invalid returns clip to 255).
    """
    depth_m = depth_u16.astype(np.float32) * scale_m
    v8 = np.clip(depth_m / max_m, 0.0, 1.0)
    v8 = np.round(v8 * 255.0).astype(np.uint8)  # [T, H, W]
    t, h, w = v8.shape
    if USE_GPU:
        # Lossless: NVENC high-throughput lossless mode (bit-exact, no qp 0 lossy)
        codec = ["-c:v", "h264_nvenc", "-preset", "lossless"]
    else:
        codec = ["-c:v", "libx264", "-qp", "0", "-preset", "medium"]
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
        "-vf", "format=yuv420p",
        *codec,
        "-an", str(dst),
    ]
    proc = subprocess.run(cmd, input=v8.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg depth encode failed for {dst}:\n{proc.stderr.decode()[-2000:]}")


def load_parquet_columns(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {name: np.array(table.column(name).to_pylist()) for name in table.schema.names}


def fk_joint_block(fk: PiperXForwardKinematics, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T, 6] joint angles -> ([T, 3] positions, [T, 9] flattened rotation matrices)."""
    positions = np.empty((joints.shape[0], 3), dtype=np.float64)
    rotmats = np.empty((joints.shape[0], 9), dtype=np.float64)
    for t in range(joints.shape[0]):
        matrix = fk.compute_fk_matrix(joints[t])
        positions[t] = matrix[:3, 3]
        rotmats[t] = matrix[:3, :3].reshape(-1)
    return positions, rotmats


def convert_episode(
    episode_index: int,
    src_root: Path,
    dst_root: Path,
    fps: float,
    instructions: list[str],
    depth_scale_m: float,
    depth_max_m: float,
    depth_dir_name: str = "depths",
) -> tuple[int, int]:
    fk = PiperXForwardKinematics()

    ep6 = f"{episode_index:06d}"
    ep7 = f"{episode_index:07d}"
    chunk_src = "chunk-000"
    chunk_dst = "chunk-0000"

    parquet_path = src_root / "data" / chunk_src / f"episode_{ep6}.parquet"
    columns = load_parquet_columns(parquet_path)
    state = np.stack(columns["observation.state"].astype(np.float64))
    action = np.stack(columns["action"].astype(np.float64))
    num_frames = state.shape[0]

    episode_key = f"{chunk_dst}/episode_{ep7}"
    data_dir = dst_root / "data" / chunk_dst
    data_dir.mkdir(parents=True, exist_ok=True)

    observations: dict[str, dict] = {}
    for camera_key, (view, cam_type) in CAMERA_VIEW_MAP.items():
        view_dir_rgb = dst_root / "video" / view / chunk_dst
        view_dir_depth = dst_root / "depth" / view / chunk_dst
        view_dir_rgb.mkdir(parents=True, exist_ok=True)
        view_dir_depth.mkdir(parents=True, exist_ok=True)

        rgb_src = src_root / "videos" / chunk_src / camera_key / f"episode_{ep6}.mp4"
        depth_src = src_root / depth_dir_name / camera_key.split(".")[-1] / f"episode-{ep6}.mkv"
        rgb_dst = view_dir_rgb / f"episode_{ep7}.mp4"
        depth_dst = view_dir_depth / f"episode_{ep7}.mp4"

        transcode_rgb(rgb_src, rgb_dst, fps)

        width, height = 640, 480
        depth_u16 = read_depth_u16(depth_src, width, height)
        if depth_u16.shape[0] != num_frames:
            raise ValueError(
                f"Depth frame count mismatch for episode {episode_index} view {view}: "
                f"{depth_u16.shape[0]} != {num_frames}"
            )
        encode_depth(depth_u16, depth_dst, fps, depth_scale_m, depth_max_m)

        observations[view] = {
            "type": cam_type,
            "rgb_path": str(rgb_dst.relative_to(dst_root)),
            "depth_path": str(depth_dst.relative_to(dst_root)),
            "start": 0,
            "end": num_frames,
            "fps": fps,
        }

    left_joints_state, left_grip_state = state[:, :NUM_JOINTS], state[:, NUM_JOINTS : NUM_JOINTS + 1]
    right_joints_state, right_grip_state = state[:, NUM_JOINTS + 1 : -1], state[:, -1:]
    left_joints_act, left_grip_act = action[:, :NUM_JOINTS], action[:, NUM_JOINTS : NUM_JOINTS + 1]
    right_joints_act, right_grip_act = action[:, NUM_JOINTS + 1 : -1], action[:, -1:]

    left_pos_p, left_rotm_p = fk_joint_block(fk, left_joints_state)
    right_pos_p, right_rotm_p = fk_joint_block(fk, right_joints_state)
    left_pos_a, left_rotm_a = fk_joint_block(fk, left_joints_act)
    right_pos_a, right_rotm_a = fk_joint_block(fk, right_joints_act)

    episode_json = {
        "num_frames": num_frames,
        "instructions": instructions,
        "observations": observations,
        "proprios": {
            "left_ee_pos": left_pos_p.tolist(),
            "left_ee_rotm": left_rotm_p.tolist(),
            "left_gripper_pos": left_grip_state.tolist(),
            "right_ee_pos": right_pos_p.tolist(),
            "right_ee_rotm": right_rotm_p.tolist(),
            "right_gripper_pos": right_grip_state.tolist(),
        },
        "actions": {
            "left_ee_pos": left_pos_a.tolist(),
            "left_ee_rotm": left_rotm_a.tolist(),
            "left_gripper_pos": left_grip_act.tolist(),
            "right_ee_pos": right_pos_a.tolist(),
            "right_ee_rotm": right_rotm_a.tolist(),
            "right_gripper_pos": right_grip_act.tolist(),
        },
    }
    with open(data_dir / f"episode_{ep7}.json", "w", encoding="utf-8") as f:
        json.dump(episode_json, f)

    return episode_index, num_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="LeRobot dataset root (raw_data/<name>).")
    parser.add_argument("--dst", type=Path, required=True, help="Output X-WAM dataset root.")
    parser.add_argument("--depth-dir-name", type=str, default="depths",
                        help="Name of depth directory inside --src (e.g., 'depths', 'depths_moge3').")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--depth-scale-m", type=float, default=0.001, help="uint16 raw -> meters scale.")
    parser.add_argument("--depth-max-m", type=float, default=1.2, help="Meters clipped to 8-bit max.")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use NVENC (h264_nvenc) for H.264 encoding instead of libx264. "
                             "Requires an NVIDIA GPU with ffmpeg nvenc support.")
    args = parser.parse_args()

    global USE_GPU
    USE_GPU = args.use_gpu
    if USE_GPU:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.04",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "h264_nvenc not available in this ffmpeg build / no usable GPU:\n"
                f"{probe.stderr.decode()[-1000:]}"
            )
        print("GPU encoding enabled (h264_nvenc)")

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset root not found: {src_root}")
    if dst_root.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {dst_root}")
    dst_root.mkdir(parents=True)

    # Tasks / instructions
    tasks = []
    with open(src_root / "meta" / "tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    task_by_index = {int(t["task_index"]): str(t["task"]) for t in tasks}

    # Episodes
    episodes = []
    with open(src_root / "meta" / "episodes.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes.sort(key=lambda e: int(e["episode_index"]))
    print(f"Converting {len(episodes)} episodes: {src_root} -> {dst_root}")
    print(f"depth-scale={args.depth_scale_m} m/raw, depth-max={args.depth_max_m} m")

    metadata: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                convert_episode,
                int(ep["episode_index"]),
                src_root,
                dst_root,
                args.fps,
                [str(t) for t in ep["tasks"]],
                args.depth_scale_m,
                args.depth_max_m,
                args.depth_dir_name,
            ): ep
            for ep in episodes
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Episodes", ncols=90):
            episode_index, num_frames = future.result()
            metadata[f"chunk-0000/episode_{episode_index:07d}"] = num_frames

    metadata = dict(sorted(metadata.items()))
    with open(dst_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    total = sum(metadata.values())
    print(f"Wrote metadata.json ({len(metadata)} episodes, {total} frames)")


if __name__ == "__main__":
    main()

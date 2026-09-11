"""Offline teacher-forcing action-fit evaluation on a raw LeRobot v2 dataset.

This program never creates a robot client and never sends commands.  For each
selected recorded observation ``(RGB[t], joint_state[t])`` it calls exactly the
same :class:`XWAMPolicy` used by the websocket server.  It reports three
separate tests: raw delta actions versus labels, sparse absolute EE proprio
nodes versus labels, and raw actions accumulated into absolute EE poses versus
the predicted sparse proprio nodes.

The raw action ground truth in the first test is the training label at every
action horizon ``h``::

    FK(action[t + h]) - FK(state[t + h])

where position is a global XYZ difference, rotation is the global rotvec of
``R_action @ R_state.T``, and gripper is ``action - state``.  This is exactly
``RobotDataset._build_delta_action_tensor`` with ``action_skip=1``.

Raw RGB is AV1 and decord cannot decode it.  Frames are therefore extracted
losslessly by the system ``ffmpeg`` decoder; no converted dataset is required.

Example (run from the X-WAM repository on the policy GPU host)::

    .venv/bin/python deployment/evaluate_raw_action_fit.py \
      --raw-dataset raw_data/place_fruits_in_bucket_v1 \
      --exp-path experiments/multitask_merged-v1-sft \
      --wan-checkpoint-dir checkpoints/Wan2.2-TI2V-5B \
      --deployment-checkpoint experiments/multitask_merged-v1-sft/checkpoints/last.deployment.pt \
      --max-queries 24 --frame-stride 300 \
      --output deployment/place_fruits_action_fit.json
"""


import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from piperx_fk import PiperXForwardKinematics  # noqa: E402


VIEW_KEYS = (
    "observation.images.fixed_front",
    "observation.images.left_arm",
    "observation.images.right_arm",
)
ACTION_NAMES = (
    "left_dx_m",
    "left_dy_m",
    "left_dz_m",
    "left_drx_rad",
    "left_dry_rad",
    "left_drz_rad",
    "left_dgripper_rad",
    "right_dx_m",
    "right_dy_m",
    "right_dz_m",
    "right_drx_rad",
    "right_dry_rad",
    "right_drz_rad",
    "right_dgripper_rad",
)
PROPRIO_NAMES = (
    "left_x_m", "left_y_m", "left_z_m", "left_qw", "left_qx", "left_qy", "left_qz", "left_gripper_rad",
    "right_x_m", "right_y_m", "right_z_m", "right_qw", "right_qx", "right_qy", "right_qz", "right_gripper_rad",
)


@dataclass(frozen=True)
class Query:
    episode_index: int
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw LeRobot v2 dataset root.")
    parser.add_argument("--exp-path", type=Path, required=True, help="Experiment directory containing config.yaml.")
    parser.add_argument("--wan-checkpoint-dir", type=Path, required=True, help="Wan2.2-TI2V-5B checkpoint directory.")
    parser.add_argument("--deployment-checkpoint", type=Path, required=True, help="Inference-only *.deployment.pt checkpoint.")
    parser.add_argument("--prompt-embeddings", type=Path, default=None, help="Optional precomputed prompt_embeddings.pt.")
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Episodes to evaluate. Omit to sample over every episode.",
    )
    parser.add_argument("--frame-stride", type=int, default=96, help="Candidate spacing in raw 30 Hz frames.")
    parser.add_argument(
        "--min-frame-fraction",
        type=float,
        default=0.1,
        help="Earliest candidate as a fraction of the episode's valid action-start range.",
    )
    parser.add_argument(
        "--max-frame-fraction",
        type=float,
        default=0.9,
        help="Latest candidate as a fraction of the episode's valid action-start range.",
    )
    parser.add_argument("--max-queries", type=int, default=24, help="Maximum sampled observations/inferences.")
    parser.add_argument("--expected-horizon", type=int, default=32, help="Only select frames with this many valid GT actions.")
    parser.add_argument("--frame-skip", type=int, default=4, help="Raw frames between predicted proprio nodes (training value: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Fixed inference seed, matching the real-robot payload default.")
    parser.add_argument(
        "--vary-seed-by-query",
        action="store_true",
        help="Use seed + query index instead of one fixed seed. Off by default to mirror deployment.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report (parent directories are created).")
    parser.add_argument(
        "--save-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include predicted and GT action chunks in the JSON report (default: true).",
    )
    args = parser.parse_args()
    if args.frame_stride < 1 or args.max_queries < 1 or args.expected_horizon < 1 or args.frame_skip < 1:
        parser.error("--frame-stride, --max-queries, --expected-horizon, and --frame-skip must be positive.")
    if not 0.0 <= args.min_frame_fraction <= args.max_frame_fraction <= 1.0:
        parser.error("Require 0 <= --min-frame-fraction <= --max-frame-fraction <= 1.")
    return args


def require_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Delay heavy imports so ``--help`` works even on a data-only machine."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read raw LeRobot parquet files. Install/sync the X-WAM environment first."
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for X-WAM inference.") from exc
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("scipy is required for training-identical rotation delta labels.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is visible to PyTorch; this deployment checkpoint requires the policy GPU.")
    return pq, torch, Rotation


def load_tasks(raw_root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    with (raw_root / "meta" / "tasks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                tasks[int(entry["task_index"])] = str(entry["task"])
    if not tasks:
        raise ValueError("No tasks found in meta/tasks.jsonl.")
    return tasks


def parquet_path(raw_root: Path, episode_index: int) -> Path:
    return raw_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def episode_indices(raw_root: Path, requested: list[int] | None) -> list[int]:
    available = sorted(int(path.stem.split("_")[-1]) for path in (raw_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not available:
        raise FileNotFoundError(f"No episode parquet files under {raw_root / 'data' / 'chunk-000'}.")
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested episodes are not present: {missing}")
    return requested


def load_episode(pq: Any, raw_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, int]:
    table = pq.read_table(parquet_path(raw_root, episode_index), columns=["observation.state", "action", "task_index"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
        raise ValueError(f"Episode {episode_index}: expected state/action [T,14], got {state.shape}/{action.shape}.")
    if np.any(task_indices != task_indices[0]):
        raise ValueError(f"Episode {episode_index}: task_index changes within an episode.")
    return state, action, int(task_indices[0])


def select_queries(
    pq: Any,
    raw_root: Path,
    episodes: list[int],
    stride: int,
    horizon: int,
    limit: int,
    min_fraction: float,
    max_fraction: float,
) -> list[Query]:
    candidates: list[Query] = []
    for episode_index in episodes:
        table = pq.read_table(parquet_path(raw_root, episode_index), columns=["frame_index"])
        num_frames = len(table)
        # The action token at h uses state/action[t+h].
        max_start = num_frames - horizon
        if max_start < 0:
            continue
        first_start = int(np.ceil(max_start * min_fraction))
        last_start = int(np.floor(max_start * max_fraction))
        for frame_index in range(first_start, last_start + 1, stride):
            candidates.append(Query(episode_index, frame_index))
    if not candidates:
        raise ValueError(
            "No valid queries in the requested frame-fraction range: episodes must contain at least "
            f"--expected-horizon={horizon} frames."
        )
    if len(candidates) <= limit:
        return candidates
    # Uniformly covering the candidate list gives every episode a chance instead
    # of spending a fixed inference budget entirely on episode 0.
    selected = np.linspace(0, len(candidates) - 1, limit, dtype=np.int64)
    return [candidates[int(i)] for i in selected]


def read_rgb_frame(raw_root: Path, episode_index: int, frame_index: int, view_key: str, image_shape: tuple[int, int, int]) -> np.ndarray:
    video = raw_root / "videos" / "chunk-000" / view_key / f"episode_{episode_index:06d}.mp4"
    height, width, channels = image_shape
    command = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select=eq(n\\,{frame_index})",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    expected_bytes = height * width * channels
    if proc.returncode != 0 or len(proc.stdout) != expected_bytes:
        raise RuntimeError(
            f"ffmpeg could not decode {video} frame {frame_index}: returncode={proc.returncode}, "
            f"bytes={len(proc.stdout)}/{expected_bytes}, stderr={proc.stderr.decode(errors='replace')[-500:]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, channels).copy()


def load_images(raw_root: Path, episode_index: int, frame_index: int, image_shape: tuple[int, int, int]) -> np.ndarray:
    return np.stack(
        [read_rgb_frame(raw_root, episode_index, frame_index, key, image_shape) for key in VIEW_KEYS], axis=0
    )


def ee_delta_labels(
    fk: PiperXForwardKinematics,
    Rotation: Any,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Return the unnormalised [T, 14] target produced by the training loader."""
    output = np.empty((len(states), 14), dtype=np.float32)
    for arm_index, offset in enumerate((0, 7)):
        for timestep, (state, action) in enumerate(zip(states, actions)):
            state_matrix = fk.compute_fk_matrix(state[offset : offset + 6])
            action_matrix = fk.compute_fk_matrix(action[offset : offset + 6])
            start = arm_index * 7
            output[timestep, start : start + 3] = action_matrix[:3, 3] - state_matrix[:3, 3]
            delta_rotation = action_matrix[:3, :3] @ state_matrix[:3, :3].T
            output[timestep, start + 3 : start + 6] = Rotation.from_matrix(delta_rotation).as_rotvec()
            output[timestep, start + 6] = action[offset + 6] - state[offset + 6]
    return output


def ee_proprio_labels(fk: PiperXForwardKinematics, Rotation: Any, states: np.ndarray) -> np.ndarray:
    """Return raw [T,16] EE proprio in exactly the training/server convention."""
    output = np.empty((len(states), 16), dtype=np.float32)
    for arm_index, offset in enumerate((0, 7)):
        start = arm_index * 8
        for timestep, state in enumerate(states):
            matrix = fk.compute_fk_matrix(state[offset : offset + 6])
            quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            if quat_wxyz[0] < 0:
                quat_wxyz *= -1
            output[timestep, start : start + 3] = matrix[:3, 3]
            output[timestep, start + 3 : start + 7] = quat_wxyz
            output[timestep, start + 7] = state[offset + 6]
    return output


def vector_cosine(prediction: np.ndarray, target: np.ndarray, threshold: float) -> tuple[float | None, int]:
    pred_norm = np.linalg.norm(prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    valid = (pred_norm > threshold) & (target_norm > threshold)
    if not np.any(valid):
        return None, 0
    cosine = np.sum(prediction[valid] * target[valid], axis=-1) / (pred_norm[valid] * target_norm[valid])
    return float(np.mean(cosine)), int(np.count_nonzero(valid))


def metric_block(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    position_pred = np.concatenate((prediction[..., 0:3], prediction[..., 7:10]), axis=-2).reshape(-1, 3)
    position_gt = np.concatenate((target[..., 0:3], target[..., 7:10]), axis=-2).reshape(-1, 3)
    rotation_pred = np.concatenate((prediction[..., 3:6], prediction[..., 10:13]), axis=-2).reshape(-1, 3)
    rotation_gt = np.concatenate((target[..., 3:6], target[..., 10:13]), axis=-2).reshape(-1, 3)
    pos_cosine, pos_count = vector_cosine(position_pred, position_gt, threshold=1e-4)
    rot_cosine, rot_count = vector_cosine(rotation_pred, rotation_gt, threshold=1e-4)
    return {
        "num_action_tokens": int(np.prod(prediction.shape[:-1])),
        "mae_per_dimension": dict(zip(ACTION_NAMES, np.mean(np.abs(error), axis=tuple(range(error.ndim - 1))).tolist())),
        "rmse_per_dimension": dict(zip(ACTION_NAMES, np.sqrt(np.mean(error**2, axis=tuple(range(error.ndim - 1)))).tolist())),
        "position_mae_m": float(np.mean(np.abs(error[..., [0, 1, 2, 7, 8, 9]]))),
        "rotation_mae_rad": float(np.mean(np.abs(error[..., [3, 4, 5, 10, 11, 12]]))),
        "gripper_mae_rad": float(np.mean(np.abs(error[..., [6, 13]]))),
        "position_vector_cosine": pos_cosine,
        "position_cosine_count": pos_count,
        "rotation_vector_cosine": rot_cosine,
        "rotation_cosine_count": rot_count,
    }


def arm_action_metrics(prediction: np.ndarray, target: np.ndarray, arm_index: int) -> dict[str, Any]:
    """Metrics for one arm's raw delta-action label, with position in millimetres."""
    offset = arm_index * 7
    pred = prediction[..., offset : offset + 7]
    gt = target[..., offset : offset + 7]
    pos_error = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)
    rot_error = np.linalg.norm(pred[..., 3:6] - gt[..., 3:6], axis=-1)
    return {
        "position_l2_mae_mm": float(np.mean(pos_error) * 1000.0),
        "position_l2_p95_mm": float(np.quantile(pos_error, 0.95) * 1000.0),
        "rotation_rotvec_l2_mae_deg": float(np.mean(rot_error) * 180.0 / np.pi),
        "gripper_mae_rad": float(np.mean(np.abs(pred[..., 6] - gt[..., 6]))),
        "predicted_position_norm_mean_mm": float(np.mean(np.linalg.norm(pred[..., :3], axis=-1)) * 1000.0),
        "target_position_norm_mean_mm": float(np.mean(np.linalg.norm(gt[..., :3], axis=-1)) * 1000.0),
    }


def proprio_arm_metrics(prediction: np.ndarray, target: np.ndarray, arm_index: int) -> dict[str, Any]:
    """Absolute-pose trajectory metrics for one arm.

    Quaternion distance uses the sign-invariant geodesic angle.  It remains
    valid even if the diffusion output chooses the opposite quaternion sign.
    """
    offset = arm_index * 8
    pred = prediction[..., offset : offset + 8].astype(np.float64)
    gt = target[..., offset : offset + 8].astype(np.float64)
    position_error = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)
    pred_quat = pred[..., 3:7]
    gt_quat = gt[..., 3:7]
    pred_norm = np.linalg.norm(pred_quat, axis=-1)
    gt_norm = np.linalg.norm(gt_quat, axis=-1)
    valid = (pred_norm > 1e-8) & (gt_norm > 1e-8)
    angle = np.full(pred_norm.shape, np.nan, dtype=np.float64)
    dot = np.sum(pred_quat[valid] * gt_quat[valid], axis=-1) / (pred_norm[valid] * gt_norm[valid])
    angle[valid] = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    pred_motion = np.linalg.norm(pred[..., :3] - pred[:, :1, :3], axis=-1)
    gt_motion = np.linalg.norm(gt[..., :3] - gt[:, :1, :3], axis=-1)
    orientation_mae_deg = float(np.mean(angle[valid]) * 180.0 / np.pi) if np.any(valid) else None
    return {
        "position_l2_mae_mm": float(np.mean(position_error) * 1000.0),
        "position_l2_p95_mm": float(np.quantile(position_error, 0.95) * 1000.0),
        "orientation_geodesic_mae_deg": orientation_mae_deg,
        "gripper_mae_rad": float(np.mean(np.abs(pred[..., 7] - gt[..., 7]))),
        "predicted_motion_from_node0_mean_mm": float(np.mean(pred_motion) * 1000.0),
        "target_motion_from_node0_mean_mm": float(np.mean(gt_motion) * 1000.0),
        "predicted_endpoint_motion_mean_mm": float(np.mean(pred_motion[:, -1]) * 1000.0),
        "target_endpoint_motion_mean_mm": float(np.mean(gt_motion[:, -1]) * 1000.0),
        "valid_quaternion_fraction": float(np.mean(valid)),
    }


def make_report(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {
        "all_horizons": metric_block(predictions, targets),
        "first_action": metric_block(predictions[:, :1], targets[:, :1]),
        "per_horizon": [],
    }
    for horizon in range(predictions.shape[1]):
        item = metric_block(predictions[:, horizon : horizon + 1], targets[:, horizon : horizon + 1])
        item["horizon_index"] = horizon
        report["per_horizon"].append(item)
    report["per_arm"] = {name: arm_action_metrics(predictions, targets, arm) for arm, name in enumerate(("left", "right"))}
    return report


def make_proprio_report(predictions: np.ndarray, targets: np.ndarray, frame_skip: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "all_nodes": {name: proprio_arm_metrics(predictions, targets, arm) for arm, name in enumerate(("left", "right"))},
        "per_node": [],
    }
    for node in range(predictions.shape[1]):
        node_metrics: dict[str, Any] = {}
        for arm, name in enumerate(("left", "right")):
            metrics = proprio_arm_metrics(predictions[:, node : node + 1], targets[:, node : node + 1], arm)
            offset = arm * 8
            predicted_motion = np.linalg.norm(
                predictions[:, node, offset : offset + 3] - predictions[:, 0, offset : offset + 3], axis=-1
            )
            target_motion = np.linalg.norm(
                targets[:, node, offset : offset + 3] - targets[:, 0, offset : offset + 3], axis=-1
            )
            # The one-node metric alone has no temporal reference, so replace
            # its zero-valued motion fields with actual node0 -> node motion.
            metrics["predicted_motion_from_node0_mean_mm"] = float(np.mean(predicted_motion) * 1000.0)
            metrics["target_motion_from_node0_mean_mm"] = float(np.mean(target_motion) * 1000.0)
            node_metrics[name] = metrics
        report["per_node"].append(
            {
                "node_index": node,
                "raw_frame_offset": node * frame_skip,
                **node_metrics,
            }
        )
    return report


def accumulate_delta_actions_to_proprios(
    initial_proprio: np.ndarray, actions: np.ndarray, Rotation: Any
) -> np.ndarray:
    """Accumulate bimanual delta actions into absolute ``[T, 16]`` EE poses.

    This is a direct two-arm port of
    ``evaluation/X-WAM/deploy_policy.py::compute_future_poses``: global
    position addition, ``dR * R`` rotation composition, wxyz quaternions, and
    additive gripper.  It intentionally does *not* apply deployment base or
    end-effector-axis transforms: both operands are in the model-native EE
    convention.
    """
    initial_proprio = np.asarray(initial_proprio, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if initial_proprio.shape != (16,) or actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(
            "Expected initial_proprio [16] and actions [T,14], got "
            f"{initial_proprio.shape} and {actions.shape}."
        )

    poses = np.empty((actions.shape[0], 16), dtype=np.float32)
    for arm_index in range(2):
        action_offset = arm_index * 7
        proprio_offset = arm_index * 8
        pos = initial_proprio[proprio_offset : proprio_offset + 3].copy()
        quat_wxyz = initial_proprio[proprio_offset + 3 : proprio_offset + 7]
        rot = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
        gripper = float(initial_proprio[proprio_offset + 7])
        for timestep, action in enumerate(actions[:, action_offset : action_offset + 7]):
            pos = pos + action[:3]
            rot = Rotation.from_rotvec(action[3:6]) * rot
            gripper = gripper + action[6]
            poses[timestep, proprio_offset : proprio_offset + 3] = pos
            poses[timestep, proprio_offset + 3 : proprio_offset + 7] = rot.as_quat()[[3, 0, 1, 2]]
            poses[timestep, proprio_offset + 7] = gripper
    return poses


def sample_accumulated_actions_at_proprio_nodes(
    initial_proprio: np.ndarray,
    actions: np.ndarray,
    frame_skip: int,
    Rotation: Any,
) -> np.ndarray:
    """Return node 0 and accumulated-action poses at raw offsets 4, 8, ... .

    Sparse node ``i`` denotes raw-frame offset ``i * frame_skip``.  It is the
    pose after applying delta index ``i * frame_skip - 1``.  Therefore 32
    actions with ``frame_skip=4`` produce nine nodes, offsets 0 through 32.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be positive.")
    accumulated = accumulate_delta_actions_to_proprios(initial_proprio, actions, Rotation)
    return np.concatenate(
        (initial_proprio[None].astype(np.float32), accumulated[frame_skip - 1 :: frame_skip]), axis=0
    )


def make_accumulated_action_vs_proprio_report(
    accumulated_action_poses: np.ndarray, sparse_proprios: np.ndarray, frame_skip: int
) -> dict[str, Any]:
    """Report disagreement of the two predicted outputs in absolute EE space."""
    if accumulated_action_poses.shape != sparse_proprios.shape:
        raise ValueError(
            "Accumulated-action and sparse-proprio trajectories must have equal shapes, got "
            f"{accumulated_action_poses.shape} and {sparse_proprios.shape}."
        )
    report = make_proprio_report(accumulated_action_poses, sparse_proprios, frame_skip)
    report["comparison"] = {
        "prediction": "absolute EE poses obtained by accumulating predicted raw delta actions",
        "reference": "predicted sparse absolute EE proprio nodes",
        "interpretation": (
            "Internal output-consistency diagnostic, not a comparison with GT. "
            "A large value means the two policy heads imply incompatible absolute trajectories."
        ),
    }
    return report


def image_shape_from_metadata(raw_root: Path) -> tuple[int, int, int]:
    with (raw_root / "meta" / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    shape = info["features"][VIEW_KEYS[0]]["shape"]
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Unexpected RGB shape in meta/info.json: {shape}")
    return tuple(int(v) for v in shape)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_root = args.raw_dataset.resolve()
    if not (raw_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a raw LeRobot dataset: {raw_root}")
    for path in (args.exp_path / "config.yaml", args.wan_checkpoint_dir, args.deployment_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    pq, _torch, Rotation = require_runtime_dependencies()
    tasks = load_tasks(raw_root)
    episodes = episode_indices(raw_root, args.episode_indices)
    queries = select_queries(
        pq,
        raw_root,
        episodes,
        args.frame_stride,
        args.expected_horizon,
        args.max_queries,
        args.min_frame_fraction,
        args.max_frame_fraction,
    )
    image_shape = image_shape_from_metadata(raw_root)
    logging.info("Selected %d offline queries over %d episodes.", len(queries), len(episodes))

    # Import only after validating data and CUDA, since instantiation allocates the policy on GPU.
    from xwam_policy import XWAMPolicy

    policy = XWAMPolicy(
        exp_path=str(args.exp_path),
        wan_checkpoint_dir=str(args.wan_checkpoint_dir),
        denoise_steps=args.denoise_steps,
        action_denoise_steps=args.action_denoise_steps,
        prompt_embeddings=None if args.prompt_embeddings is None else str(args.prompt_embeddings),
        deployment_checkpoint=str(args.deployment_checkpoint),
    )
    fk = PiperXForwardKinematics()
    cached_episodes: dict[int, tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]] = {}
    predicted_action_chunks: list[np.ndarray] = []
    target_action_chunks: list[np.ndarray] = []
    predicted_proprio_chunks: list[np.ndarray] = []
    target_proprio_chunks: list[np.ndarray] = []
    accumulated_action_proprio_chunks: list[np.ndarray] = []
    query_records: list[dict[str, Any]] = []

    for query_number, query in enumerate(queries):
        if query.episode_index not in cached_episodes:
            states, actions, task_index = load_episode(pq, raw_root, query.episode_index)
            if task_index not in tasks:
                raise ValueError(f"Episode {query.episode_index}: unknown task_index {task_index}.")
            cached_episodes[query.episode_index] = (
                states,
                actions,
                task_index,
                ee_delta_labels(fk, Rotation, states, actions),
                ee_proprio_labels(fk, Rotation, states),
            )
        states, _actions, task_index, action_labels, proprio_labels = cached_episodes[query.episode_index]
        images = load_images(raw_root, query.episode_index, query.frame_index, image_shape)
        seed = args.seed + query_number if args.vary_seed_by_query else args.seed
        result = policy.infer({"images": images, "agent_pos": states[query.frame_index], "prompt": tasks[task_index], "seed": seed})
        action_prediction = np.asarray(result["actions"], dtype=np.float32)
        action_available = len(action_labels) - query.frame_index
        action_horizon = min(len(action_prediction), action_available)
        if action_horizon < args.expected_horizon:
            raise RuntimeError(f"Query {query} unexpectedly has only {action_horizon} target action steps.")
        action_prediction = action_prediction[:action_horizon]
        action_target = action_labels[query.frame_index : query.frame_index + action_horizon]

        proprio_prediction = np.asarray(result["proprios"], dtype=np.float32)
        if proprio_prediction.ndim != 2 or proprio_prediction.shape[1] != 16:
            raise RuntimeError(f"Query {query}: expected server propriors [Tp,16], got {proprio_prediction.shape}.")
        proprio_available = 1 + (len(proprio_labels) - 1 - query.frame_index) // args.frame_skip
        proprio_horizon = min(len(proprio_prediction), proprio_available)
        if proprio_horizon < 2:
            raise RuntimeError(f"Query {query}: fewer than two valid proprio nodes.")
        proprio_prediction = proprio_prediction[:proprio_horizon]
        proprio_target = proprio_labels[
            query.frame_index : query.frame_index + proprio_horizon * args.frame_skip : args.frame_skip
        ]
        if len(proprio_target) != proprio_horizon:
            raise RuntimeError(f"Query {query}: proprio target indexing mismatch.")

        # Match ``compute_future_poses`` in the evaluation reference exactly,
        # then take the same 0, 4, ..., 32 raw-frame nodes as sparse proprio.
        accumulated_action_proprios = sample_accumulated_actions_at_proprio_nodes(
            proprio_labels[query.frame_index], action_prediction, args.frame_skip, Rotation
        )
        common_output_nodes = min(len(accumulated_action_proprios), proprio_horizon)
        if common_output_nodes < 2:
            raise RuntimeError(f"Query {query}: fewer than two action/proprio comparison nodes.")
        accumulated_action_proprios = accumulated_action_proprios[:common_output_nodes]
        sparse_proprio_for_comparison = proprio_prediction[:common_output_nodes]

        predicted_action_chunks.append(action_prediction)
        target_action_chunks.append(action_target)
        predicted_proprio_chunks.append(proprio_prediction)
        target_proprio_chunks.append(proprio_target)
        accumulated_action_proprio_chunks.append(accumulated_action_proprios)
        record: dict[str, Any] = {
            **asdict(query),
            "prompt": tasks[task_index],
            "seed": seed,
            "infer_time_s": float(result["infer_time_s"]),
            "action_horizon": action_horizon,
            "proprio_horizon": proprio_horizon,
            "tests": {
                "delta_action_vs_gt": metric_block(action_prediction[None], action_target[None]),
                "sparse_proprio_vs_gt": {
                    name: proprio_arm_metrics(proprio_prediction[None], proprio_target[None], arm)
                    for arm, name in enumerate(("left", "right"))
                },
                "accumulated_absolute_pose_vs_sparse_proprio": {
                    name: proprio_arm_metrics(accumulated_action_proprios[None], sparse_proprio_for_comparison[None], arm)
                    for arm, name in enumerate(("left", "right"))
                },
            },
        }
        if args.save_actions:
            record["predicted_actions"] = action_prediction.tolist()
            record["ground_truth_actions"] = action_target.tolist()
            record["predicted_proprios"] = proprio_prediction.tolist()
            record["ground_truth_proprios"] = proprio_target.tolist()
            record["accumulated_action_proprios"] = accumulated_action_proprios.tolist()
        query_records.append(record)
        logging.info(
            "[%d/%d] ep=%06d frame=%d: action pos MAE=%.4f m; proprio L/R pos MAE=%.1f/%.1f mm",
            query_number + 1, len(queries), query.episode_index, query.frame_index,
            record["tests"]["delta_action_vs_gt"]["position_mae_m"],
            record["tests"]["sparse_proprio_vs_gt"]["left"]["position_l2_mae_mm"],
            record["tests"]["sparse_proprio_vs_gt"]["right"]["position_l2_mae_mm"],
        )

    # Every query was selected to have expected_horizon valid labels.  Requiring a
    # common horizon makes the per-horizon aggregation unambiguous.
    common_action_horizon = min(chunk.shape[0] for chunk in predicted_action_chunks)
    action_predictions = np.stack([chunk[:common_action_horizon] for chunk in predicted_action_chunks])
    action_targets = np.stack([chunk[:common_action_horizon] for chunk in target_action_chunks])
    common_proprio_horizon = min(chunk.shape[0] for chunk in predicted_proprio_chunks)
    proprio_predictions = np.stack([chunk[:common_proprio_horizon] for chunk in predicted_proprio_chunks])
    proprio_targets = np.stack([chunk[:common_proprio_horizon] for chunk in target_proprio_chunks])
    common_comparison_horizon = min(chunk.shape[0] for chunk in accumulated_action_proprio_chunks)
    accumulated_action_proprios = np.stack(
        [chunk[:common_comparison_horizon] for chunk in accumulated_action_proprio_chunks]
    )
    sparse_proprios_for_comparison = np.stack(
        [chunk[:common_comparison_horizon] for chunk in predicted_proprio_chunks]
    )
    delta_action_vs_gt = make_report(action_predictions, action_targets)
    sparse_proprio_vs_gt = make_proprio_report(proprio_predictions, proprio_targets, args.frame_skip)
    accumulated_absolute_pose_vs_sparse_proprio = make_accumulated_action_vs_proprio_report(
        accumulated_action_proprios, sparse_proprios_for_comparison, args.frame_skip
    )
    report = {
        "definition": {
            "mode": "recorded-current-observation, open-loop diffusion trajectory evaluation",
            "input": "RGB[t], recorded joint state[t], recorded prompt",
            "delta_action_ground_truth": (
                "FK(action[t+h]) - FK(state[t+h]), matching "
                "RobotDataset._build_delta_action_tensor(action_skip=1)"
            ),
            "action_names": list(ACTION_NAMES),
            "proprio_ground_truth": "FK(state[t + node * frame_skip]), matching RobotDataset._build_proprio_tensor",
            "proprio_names": list(PROPRIO_NAMES),
            "view_order": ["fixed_front", "left_arm", "right_arm"],
            "tests": {
                "action_metrics": "Predicted raw [T,14] delta-action labels versus GT at every raw step.",
                "proprio_metrics": "Predicted sparse absolute [Tp,16] EE proprio nodes versus GT.",
                "accumulated_action_vs_sparse_proprio": (
                    "Predicted actions accumulated by evaluation/X-WAM/deploy_policy.py "
                    "semantics, sampled at sparse nodes, versus predicted sparse propriors."
                ),
            },
        },
        "run": {
            "raw_dataset": str(raw_root),
            "exp_path": str(args.exp_path.resolve()),
            "deployment_checkpoint": str(args.deployment_checkpoint.resolve()),
            "denoise_steps": args.denoise_steps,
            "action_denoise_steps": args.action_denoise_steps,
            "seed": args.seed,
            "vary_seed_by_query": args.vary_seed_by_query,
            "frame_skip": args.frame_skip,
            "min_frame_fraction": args.min_frame_fraction,
            "max_frame_fraction": args.max_frame_fraction,
            "num_queries": len(query_records),
            "common_action_horizon": common_action_horizon,
            "common_proprio_horizon": common_proprio_horizon,
            "common_accumulated_action_vs_sparse_proprio_horizon": common_comparison_horizon,
        },
        "tests": {
            "delta_action_vs_gt": delta_action_vs_gt,
            "sparse_proprio_vs_gt": sparse_proprio_vs_gt,
            "accumulated_absolute_pose_vs_sparse_proprio": accumulated_absolute_pose_vs_sparse_proprio,
        },
        "queries": query_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "delta_action_vs_gt_first_action": report["tests"]["delta_action_vs_gt"]["first_action"],
                "sparse_proprio_vs_gt_all_nodes": report["tests"]["sparse_proprio_vs_gt"]["all_nodes"],
                "accumulated_absolute_pose_vs_sparse_proprio_all_nodes": report["tests"][
                    "accumulated_absolute_pose_vs_sparse_proprio"
                ]["all_nodes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

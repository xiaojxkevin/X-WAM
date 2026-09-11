"""X-WAM inference policy for real-robot serving.

Wraps ``XWAMRunner`` behind a simple ``infer(payload) -> result`` interface, mirroring the
official ``evaluation/policy_server.py`` preprocessing but exposed as a callable object so it
can be served over any transport (websocket / zmq / ...).

Payload (a single python dict, numpy arrays encoded via msgpack on the wire):

    {
        # RGB: [V, H, W, 3] uint8, V=3 in ALPHABETICAL view order
        # (fixed_front, left_arm, right_arm) -- matches training view ordering.
        "images": np.uint8 [V, H, W, 3],

        # EITHER raw 16-dim end-effector state:
        "proprios": np.float32 [16],
        #   [left_ee_xyz(3), left_quat_wxyz(4), left_gripper(1),
        #    right_ee_xyz(3), right_quat_wxyz(4), right_gripper(1)]
        #   in RAW units (meters / canonical quat / raw jaw radians), i.e. the same
        #   convention as ``proprios`` in the training episode JSONs.
        #
        # OR raw 14-dim joint state, FK is applied server-side with scripts/piperx_fk.py:
        "agent_pos": np.float32 [14],
        #   [left_j1..j6, left_gripper, right_j1..j6, right_gripper] (radians)

        "prompt": str,
        "seed": int (optional, deterministic sampling),
        "cfg": float (optional, default 0.0 = no classifier-free guidance),
    }

Result:

    {
        "actions": np.float32 [Ta, 14],  # delta EE correction per step at action_fps:
                                         # [l_dxyz(3), l_drotvec(3), l_dgrip(1), r_...]
                                         # relative to the state/proprio at that same
                                         # step (all global frame; do not accumulate).
        "proprios": np.float32 [Tp, 16], # predicted absolute EE chain (video-frame rate),
        "action_fps": float, "proprio_fps": float,
        "view_order": [...], "prompt": str, "seed": int,
    }

All returned poses are in the SAME robot-side convention as the input proprio.
"""

import logging
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from runners.xwam_runner import XWAMRunner
from piperx_fk import PiperXForwardKinematics

DIMS_PER_ARM = 7
PROPRIO_DIM = 16


def resize_and_center_crop(tensor: torch.Tensor, resized_shape: tuple[int, int], crop_ratio: float) -> torch.Tensor:
    """Official preprocessing: resize -> center crop (crop_ratio) -> resize back. [B, V, C, H, W]."""
    B, V = tensor.shape[:2]
    tensor = tensor.flatten(0, 1)
    tensor = TF.resize(tensor, size=list(resized_shape), interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    tensor = tensor.unflatten(0, (B, V))
    H, W = resized_shape
    crop_h, crop_w = int(H * crop_ratio), int(W * crop_ratio)
    top, left = (H - crop_h) // 2, (W - crop_w) // 2
    tensor = tensor[:, :, :, top : top + crop_h, left : left + crop_w]
    tensor = tensor.flatten(0, 1)
    tensor = TF.resize(
        tensor, size=list(resized_shape), interpolation=TF.InterpolationMode.BILINEAR, antialias=False
    )
    return tensor.unflatten(0, (B, V))


def _rotm_to_canonical_quat_wxyz(rotm: np.ndarray) -> np.ndarray:
    """Rotation matrices [3, 3] -> canonical quaternion wxyz [4] with positive w (matches training)."""
    quat_xyzw = Rotation.from_matrix(rotm).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    if quat_wxyz[0] < 0:
        quat_wxyz *= -1
    return quat_wxyz


class XWAMPolicy:
    """Loads the X-WAM checkpoint and answers ``infer`` requests."""

    def __init__(
        self,
        exp_path: str,
        steps: str = "last",
        wan_checkpoint_dir: str | None = None,
        denoise_steps: int = 50,
        action_denoise_steps: int = 10,
        device: str = "cuda",
        profile: bool = False,
        prompt_embeddings: str | None = None,
        deployment_checkpoint: str | None = None,
    ):
        self.exp_path = exp_path
        config = OmegaConf.load(os.path.join(exp_path, "config.yaml"))
        config.sample_steps = denoise_steps
        config.use_decoupled_inference = action_denoise_steps > 0
        config.action_denoise_steps = action_denoise_steps
        config.action_num = config.dataset.frame_skip // config.dataset.action_skip
        # The real-robot endpoint returns actions/proprios only. Avoid constructing
        # the checkpoint's unused depth blocks so they never occupy GPU memory.
        config.use_depth = False

        if wan_checkpoint_dir is not None:
            config.wan_checkpoint_dir = wan_checkpoint_dir
        if config.get("wan_checkpoint_dir") is None:
            raise ValueError("wan_checkpoint_dir must be set in config or via --wan_checkpoint_dir")

        self.config = config
        self.device = device
        self.video_size = tuple(config.dataset.video_size)
        self.crop_ratio = float(config.dataset.crop_ratio)
        self.action_skip = int(config.dataset.action_skip)
        self.frame_skip = int(config.dataset.frame_skip)
        self.raw_fps = float(getattr(config.dataset, "fps", 30.0))
        self.view_order = None  # learned from the first request, then fixed
        self.profile = profile

        self.state_q01, self.state_q99, self.action_q01, self.action_q99 = self._build_statistics(config)

        # Pre-encoded T5 prompt embeddings: when provided, T5 is never loaded
        # (saves ~11 GB) and "prompt" must exactly match one of the encoded tasks.
        self.prompt_to_embedding = None
        self.tasks: list[str] = []
        emb_path = prompt_embeddings or os.path.join(exp_path, "prompt_embeddings.pt")
        if emb_path and os.path.exists(emb_path):
            blob = torch.load(emb_path, map_location="cpu")
            self.tasks = list(blob["prompts"])
            self._prompt_embeddings = blob["embeddings"]  # [N, L, D] bf16, kept on CPU
            self.prompt_to_embedding = {p: i for i, p in enumerate(self.tasks)}
            logging.info(
                "Loaded prompt embeddings for %d tasks from %s (T5 will be skipped).", len(self.tasks), emb_path
            )

        logging.info(f"Loading X-WAM policy from {exp_path} (steps={steps}) ...")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        logging.info("Enabled TF32 Tensor Cores for FP32 matmul (torch float32_matmul_precision=high).")

        if deployment_checkpoint is not None:
            checkpoint_candidates = (deployment_checkpoint,)
        else:
            checkpoint_root = os.path.join(exp_path, f"checkpoints/{steps}.ckpt")
            checkpoint_candidates = (
                os.path.join(checkpoint_root, "checkpoint", "mp_rank_00_model_states.pt"),
                os.path.join(checkpoint_root, "checkpoints", "mp_rank_00_model_states.pt"),
            )
        ckpt_path = next((path for path in checkpoint_candidates if os.path.isfile(path)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                "Checkpoint not found. Looked for: " + ", ".join(checkpoint_candidates)
            )

        # A deployment checkpoint is complete for no-depth serving: construct
        # DiT and VAE architecture on meta, then assign its tensors directly
        # to CUDA. This avoids loading either the base DiT safetensors or the
        # base VAE checkpoint first.
        deployment_state_only = deployment_checkpoint is not None
        self.model = XWAMRunner(
            config=config,
            skip_text_encoder=self.prompt_to_embedding is not None,
            deployment_state_only=deployment_state_only,
        )
        self.model.vae.dtype = torch.bfloat16
        # mmap prevents the 37 GB DeepSpeed state from being copied wholesale
        # into anonymous CPU memory.  load_state_dict only faults the weights
        # required by the no-depth runner.
        checkpoint_device = device if deployment_state_only else "cpu"
        ckpt = torch.load(ckpt_path, map_location=checkpoint_device, weights_only=True, mmap=True)
        incompatible = self.model.load_state_dict(
            ckpt["module"], strict=False, assign=deployment_state_only
        )
        incompatible_missing = incompatible.missing_keys
        incompatible_unexpected = incompatible.unexpected_keys
        del ckpt
        allowed_unexpected_prefixes = (
            "model.extra_blocks.",
            "model.extra_heads.",
            "text_encoder.",  # T5 skipped when pre-encoded prompt embeddings are used
        )
        invalid_unexpected = [
            key for key in incompatible_unexpected if not key.startswith(allowed_unexpected_prefixes)
        ]
        if incompatible_missing or invalid_unexpected:
            raise RuntimeError(
                "Checkpoint mismatch outside the disabled depth branch: "
                f"missing={incompatible_missing}, unexpected={invalid_unexpected}"
            )
        logging.info("Skipped %d unused checkpoint tensors.", len(incompatible_unexpected))
        if deployment_state_only:
            non_cuda_parameters = [name for name, param in self.model.named_parameters() if not param.is_cuda]
            if non_cuda_parameters:
                raise RuntimeError(
                    "Deployment checkpoint did not assign all model weights to CUDA; "
                    f"first non-CUDA key: {non_cuda_parameters[0]}"
                )
        # ``assign=True`` puts deployment tensors directly on CUDA, but it
        # bypasses LightningModule.to(), leaving its private ``_device`` at
        # the CPU default. XWAMRunner uses ``self.device`` to create the
        # sampling masks/noise, which then conflicts with VAE latents on CUDA.
        # Calling .to(cuda) here is storage-preserving for tensors already on
        # CUDA; it only updates Lightning's device bookkeeping (and any small
        # residual CPU buffers).
        self.model = self.model.to(device)
        self.model.eval()
        torch.cuda.empty_cache()
        logging.info("Enabling torch.compile for the BF16 DiT; compilation occurs during warmup.")
        self.model.model = torch.compile(self.model.model)
        self._fk = PiperXForwardKinematics()
        self._warmup_compiled_model()
        logging.info("X-WAM policy ready.")

    def _warmup_inputs(self) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
        """Fixed post-preprocessing tensors used to compile the serving shape."""
        height, width = self.video_size
        rgb = torch.zeros((1, 3, 3, height, width), device=self.device, dtype=torch.bfloat16)
        proprio = torch.zeros((1, PROPRIO_DIM), device=self.device, dtype=torch.bfloat16)
        if self.prompt_to_embedding is not None:
            if not self.tasks:
                raise RuntimeError("torch.compile warmup requires at least one pre-encoded prompt embedding.")
            return rgb, proprio, [self.tasks[0]], self._prompt_embeddings[:1].to(self.device, dtype=torch.bfloat16)
        return rgb, proprio, [""], None

    @torch.inference_mode()
    def _warmup_compiled_model(self) -> None:
        """Compile and validate the normal ten-step serving path before startup."""
        logging.info("Starting two-pass torch.compile warmup for the deployment shape ...")
        start = time.perf_counter()
        rgb, proprio, prompt, prompt_embedding = self._warmup_inputs()
        for seed in (0, 1):
            _, actions, proprios, _ = self.model.generate(
                rgb,
                proprio,
                prompt,
                seeds=[seed],
                early_stop=True,
                cfg=0.0,
                run_depth=False,
                prompt_embedding=prompt_embedding,
            )
            if not torch.isfinite(actions).all() or not torch.isfinite(proprios).all():
                raise RuntimeError("torch.compile warmup produced non-finite actions or proprios.")
        torch.cuda.synchronize(self.device)
        logging.info("torch.compile warmup finished in %.2fs.", time.perf_counter() - start)

    @staticmethod
    def _build_statistics(config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Quantile arrays from config.dataset.statistics (mirrors evaluation/policy_server.py).

        State  [16]: [left_xyz(3), left_quat(4)=identity, left_grip(1), right_...]
        Action [14]: [left_xyz(3), left_aa(3), left_grip(1), right_...]
        """
        stats = config.dataset.statistics
        state_q01 = list(stats.q01.proprio_left_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
        state_q99 = list(stats.q99.proprio_left_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
        state_q01 += list(stats.q01.proprio_right_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
        state_q99 += list(stats.q99.proprio_right_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)

        action_q01 = (
            list(stats.q01.action_left_ee_xyz)
            + list(stats.q01.action_left_ee_axisangle)
            + list(stats.q01.gripper_action)
            + list(stats.q01.action_right_ee_xyz)
            + list(stats.q01.action_right_ee_axisangle)
            + list(stats.q01.gripper_action)
        )
        action_q99 = (
            list(stats.q99.action_left_ee_xyz)
            + list(stats.q99.action_left_ee_axisangle)
            + list(stats.q99.gripper_action)
            + list(stats.q99.action_right_ee_xyz)
            + list(stats.q99.action_right_ee_axisangle)
            + list(stats.q99.gripper_action)
        )
        return np.array(state_q01), np.array(state_q99), np.array(action_q01), np.array(action_q99)

    # ------------------------------------------------------------------ #
    # Observation preprocessing
    # ------------------------------------------------------------------ #

    def _joints_to_proprio(self, agent_pos: np.ndarray) -> np.ndarray:
        """[14] raw joints -> [16] raw EE proprio via Piper-X FK (same FK as training conversion)."""
        agent_pos = np.asarray(agent_pos, dtype=np.float64)
        blocks = []
        for arm_idx in range(2):
            off = arm_idx * DIMS_PER_ARM
            joints, gripper = agent_pos[off : off + 6], agent_pos[off + 6 : off + 7]
            mat = self._fk.compute_fk_matrix(joints)
            quat = _rotm_to_canonical_quat_wxyz(mat[:3, :3])
            blocks.append(np.concatenate([mat[:3, 3], quat, gripper]))
        return np.concatenate(blocks).astype(np.float32)

    def _parse_observation(self, payload: dict) -> tuple[torch.Tensor, torch.Tensor, str, int, float]:
        images = np.asarray(payload["images"])
        if images.ndim == 4 and images.shape[0] != 3:
            raise ValueError(f"Expected 3 views in alphabetical order, got {images.shape[0]}.")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"images must be [V, H, W, 3], got {images.shape}.")

        # OpenPI's MessagePack decoder deliberately creates zero-copy ndarray
        # views over an immutable bytes payload. ``torch.from_numpy`` warns for
        # such arrays even though this preprocessing never mutates the source.
        # Copy only in that case to make the ownership explicit and safe.
        if not images.flags.writeable:
            images = images.copy()
        rgb = torch.from_numpy(images).float().unsqueeze(0)  # [1, V, H, W, 3]
        rgb = rgb.permute(0, 1, 4, 2, 3)  # -> [B, V, C, H, W]
        rgb = resize_and_center_crop(rgb, self.video_size, self.crop_ratio)
        rgb = rgb / 127.5 - 1.0
        rgb = rgb.to(self.device, dtype=torch.bfloat16)

        if "proprios" in payload and payload["proprios"] is not None:
            proprio_raw = np.asarray(payload["proprios"], dtype=np.float32)
        elif "agent_pos" in payload and payload["agent_pos"] is not None:
            proprio_raw = self._joints_to_proprio(np.asarray(payload["agent_pos"], dtype=np.float32))
        else:
            raise ValueError("payload must contain either 'proprios' [16] or 'agent_pos' [14].")
        if proprio_raw.shape != (PROPRIO_DIM,):
            raise ValueError(f"proprios must be [{PROPRIO_DIM}], got {proprio_raw.shape}.")

        proprio_norm = 2.0 * (proprio_raw - self.state_q01) / (self.state_q99 - self.state_q01) - 1.0
        proprio = (
            torch.from_numpy(proprio_norm.astype(np.float32)).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        )

        prompt = payload.get("prompt", "")
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.prompt_to_embedding is not None:
            if len(prompt) != 1 or prompt[0] not in self.prompt_to_embedding:
                raise ValueError(
                    f"prompt must be exactly one of the pre-encoded tasks: {self.tasks} (got {prompt!r})"
                )
            idx = self.prompt_to_embedding[prompt[0]]
            prompt_embedding = self._prompt_embeddings[idx : idx + 1].to(self.device, dtype=torch.bfloat16)
        else:
            prompt_embedding = None

        seed = int(payload.get("seed", np.random.randint(0, 2**31 - 1)))
        requested_cfg = float(payload.get("cfg", 0.0))
        if requested_cfg != 0.0:
            logging.warning("Ignoring cfg=%s; the memory-optimized policy fixes CFG at 0.0.", requested_cfg)
        cfg = 0.0
        return rgb, proprio, prompt, seed, cfg, prompt_embedding

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def infer(self, payload: dict) -> dict:
        t0 = time.time()
        rgb, proprio, prompt, seed, cfg, prompt_embedding = self._parse_observation(payload)
        if self.profile:
            # Synchronize before resetting so the reported peak belongs only to
            # this request's generation, not startup/warmup work on the stream.
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

        _, xt_actions, xt_proprios, _ = self.model.generate(
            rgb,
            proprio,
            prompt,
            seeds=[seed],
            early_stop=True,
            cfg=cfg,
            run_depth=False,
            prompt_embedding=prompt_embedding,
        )
        if not torch.isfinite(xt_actions).all() or not torch.isfinite(xt_proprios).all():
            raise RuntimeError("torch.compile inference produced non-finite actions or proprios.")
        inference_backend = "compile"

        if self.profile:
            gpu_end.record()

        actions_np = xt_actions[0].float().cpu().numpy()
        proprios_np = xt_proprios[0].float().cpu().numpy()
        actions_np = (actions_np[:, : len(self.action_q01)] + 1) / 2 * (self.action_q99 - self.action_q01) + self.action_q01
        proprios_np = (proprios_np + 1) / 2 * (self.state_q99 - self.state_q01) + self.state_q01

        result = {
            "actions": actions_np.astype(np.float32),
            "proprios": proprios_np.astype(np.float32),
            "action_fps": self.raw_fps / self.action_skip,
            "proprio_fps": self.raw_fps / self.frame_skip,
            "view_order": ["fixed_front", "left_arm", "right_arm"],
            "prompt": prompt[0] if prompt else "",
            "seed": seed,
            "inference_backend": inference_backend,
            "infer_time_s": time.time() - t0,
        }
        if self.profile:
            # The blocking D2H conversion above synchronizes the stream, so
            # elapsed_time and peak statistics are valid at this point.
            result["profile"] = {
                "gpu_generate_ms": float(gpu_start.elapsed_time(gpu_end)),
                "allocated_mib": round(torch.cuda.memory_allocated(self.device) / 2**20, 1),
                "reserved_mib": round(torch.cuda.memory_reserved(self.device) / 2**20, 1),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(self.device) / 2**20, 1),
            }
        logging.info(
            f"infer: {result['infer_time_s']:.2f}s | actions {result['actions'].shape} "
            f"| proprios {result['proprios'].shape} | backend={inference_backend} cfg={cfg} seed={seed}"
        )
        if self.profile:
            logging.info("GPU profile: %s", result["profile"])
        return result

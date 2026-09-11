# X-WAM Real-Robot Deployment

OpenPI-style WebSocket server serving an X-WAM SFT checkpoint for real-robot
control. One binary websocket frame per request; payload/result are python
dicts serialized with msgpack plus the bundled OpenPI-compatible NumPy codec
(numpy arrays travel inline).

## Server components

| File | Purpose |
|---|---|
| `websocket_policy_server.py` | WebSocket server (openpi-compatible wire protocol) |
| `xwam_policy.py` | Inference wrapper: preprocessing, checkpoint loading, denormalization |
| `export_deployment_checkpoint.py` | One-time DeepSpeed checkpoint trimming for real-robot serving |
| `precompute_prompt_embeddings.py` | Offline T5 prompt-embedding pre-encoding (small-GPU deployment) |
| `evaluate_raw_action_fit.py` | GPU-only teacher-forcing test: delta action vs GT, sparse proprio vs GT, and accumulated action pose vs sparse proprio |
| `lerobot_client_example.py` | Minimal client example (no `openpi_client` dependency) |

## What the server loads

- `<exp-path>/config.yaml` — model/data config, quantile statistics (q01/q99),
  `video_size`, `crop_ratio`, `fps`, `frame_skip`, `action_skip`
- `<exp-path>/checkpoints/<steps>.ckpt/checkpoints/mp_rank_00_model_states.pt` —
  DeepSpeed ZeRO model weights. It is memory-mapped at startup; only tensors
  needed by the no-depth model are faulted into RAM.
- `wan_checkpoint_dir` (default from config, e.g. `./checkpoints/Wan2.2-TI2V-5B/`) —
  VAE (`Wan2.2_VAE.pth`), DiT architecture + base weights
  (`diffusion_pytorch_model-*.safetensors`)
- `<exp-path>/prompt_embeddings.pt` (optional, auto-detected; override with
  `--prompt-embeddings`) — pre-encoded T5 prompt embeddings. When present, the
  11 GB T5 encoder is **not loaded** and `prompt` must be exactly one of the
  encoded task strings. Peak VRAM ~12 GB (fits a 24 GB RTX 3090).
- `<exp-path>/checkpoints/<steps>.deployment.pt` (generated above) is the
  preferred serving input. It contains the no-depth DiT and VAE weights, so
  the server builds their architecture on meta and assigns this checkpoint
  directly to CUDA; it does not first read the base DiT safetensors or
  `Wan2.2_VAE.pth`.

## Running

Ensure the X-WAM environment has been synchronized first; deployment additionally
requires `msgpack` and `websockets` for the WebSocket wire protocol:

```bash
uv lock && uv sync
```

```bash
# 1. (one-time) create the inference-only checkpoint (CPU only; no GPU required)
.venv/bin/python deployment/export_deployment_checkpoint.py \
    --source experiments/multitask_merged-v1-sft/checkpoints/last.ckpt/checkpoints/mp_rank_00_model_states.pt \
    --destination experiments/multitask_merged-v1-sft/checkpoints/last.deployment.pt

# 2. (one-time) pre-encode the task prompts
.venv/bin/python deployment/precompute_prompt_embeddings.py \
    --tasks-jsonl raw_data/multitask_merged_v1/meta/tasks.jsonl \
    --wan-checkpoint-dir ./checkpoints/Wan2.2-TI2V-5B \
    --config experiments/multitask_merged-v1-sft/config.yaml \
    --dst experiments/multitask_merged-v1-sft

# 3. start the server
.venv/bin/python deployment/websocket_policy_server.py \
    --exp-path experiments/multitask_merged-v1-sft \
    --host 0.0.0.0 --port 8080
```

The equivalent wrapper, which fixes the current experiment/base-weight defaults, is:

```bash
bash deployment/serve_xwam.sh
```

`serve_xwam.sh` always uses the accelerated deployment path: full-DiT
`torch.compile` plus two fixed-shape warmup passes through the default 5-step
DiT + UniPC action/proprio denoise loop. The sampling algorithm and scheduler
are unchanged.

This path requires CUDA and the auto-detected `prompt_embeddings.pt`. There is
no eager fallback: compilation, warmup, or a non-finite warmup output causes
server startup to fail. The response and `get_config` metadata report
`inference_backend: "compile"`.

For a VRAM baseline on a deployment GPU, add `PROFILE=1`.  Each response and
the server log then include GPU generation time plus allocated, reserved, and
per-request peak VRAM in MiB:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE=1 bash deployment/serve_xwam.sh
```

Warmup runs under `torch.inference_mode()`. This is required
for a 24 GB RTX 3090: otherwise PyTorch retains training autograd activations
for the full multi-step denoising warmup and can OOM.

## Wire protocol

Immediately after the WebSocket handshake, the server sends one binary msgpack
frame: `{"metadata": ...}`. This is required by OpenPI's
`WebsocketClientPolicy`. One binary frame per inference request follows.

Control commands (mainly useful for clients that reconnect or need to refresh metadata):

- `{"command": "get_config"}` → server metadata (see below)
- `{"command": "ping"}` → `{"pong": <unix ts>}`

Otherwise the dict is an observation.

### Observation (request)

| Field | Type / Shape | Notes |
|---|---|---|
| `images` | `uint8 [V=3, H, W, 3]` | **Required.** 3 camera views in alphabetical order: `fixed_front`, `left_arm`, `right_arm` (matches training view ordering). Server resizes to `video_size` then center-crops with `crop_ratio`. |
| `proprios` | `float32 [16]` | **Optional** (either this or `agent_pos`). Raw EE state: `[left_xyz(3), left_quat_wxyz(4), left_gripper(1), right_xyz(3), right_quat_wxyz(4), right_gripper(1)]` — meters, canonical quat (positive w), raw jaw radians. |
| `agent_pos` | `float32 [14]` | Alternative to `proprios`. Raw joints `[left_j1..j6, left_gripper, right_j1..j6, right_gripper]` (radians); server applies Piper-X FK (same FK as training data conversion). |
| `prompt` | `str` | Task instruction. If `prompt_embeddings.pt` is loaded, must **exactly** match one of the encoded task strings, otherwise the request is rejected with an error. |
| `seed` | `int` | Optional. Deterministic noise; defaults to random. |
| `cfg` | `float` | Optional. Currently fixed at 0.0 (no classifier-free guidance); nonzero values are ignored with a warning. |

### Result (response)

| Field | Type / Shape | Notes |
|---|---|---|
| `actions` | `float32 [Ta, 14]` | Delta end-effector corrections at `action_fps` Hz: per step `[l_dxyz(3), l_drotvec(3), l_dgrip(1), r_...]`, relative to the proprio/state at the **same time step** (global frame). Compose each action with its time-aligned proprio; do not accumulate it across steps. |
| `proprios` | `float32 [Tp, 16]` | Predicted absolute EE chain at `proprio_fps` Hz, same convention as input `proprios`. |
| `action_fps` | `float` | `raw_fps / action_skip` (e.g. 30/1 = 30 Hz with this checkpoint's `action_skip=1`... see `get_config` for the live value). |
| `proprio_fps` | `float` | `raw_fps / frame_skip` (e.g. 30/4 = 7.5 Hz). |
| `view_order` | `list[str]` | The view order the server expects (`["fixed_front", "left_arm", "right_arm"]`). |
| `prompt` / `seed` | `str` / `int` | Echo of the request (effective values). |
| `infer_time_s` | `float` | Server-side wall-clock inference time. |
| `error` | `str` | Present only on failure (bad prompt, bad shapes, ...). |

For `multitask_merged-v1-sft`: `Ta = 32` (i.e. `(frame_num-1) * frame_skip / action_skip`
= 8*4), `Tp = 9`.

## Offline action-fit replay

This is a read-only, GPU-only diagnostic; it does not start a WebSocket server
or communicate with the robot. It samples recorded ``RGB[t]`` and joint state
from a raw LeRobot dataset, runs the same `XWAMPolicy` as deployment, and
emits exactly three tests:

1. `delta_action_vs_gt`: each raw delta-action token against the exact
   unnormalised training label, `FK(action[t+h]) - FK(state[t+h])`;
2. `sparse_proprio_vs_gt`: each predicted absolute sparse EE proprio node
   against `FK(state[t + node * frame_skip])`; and
3. `accumulated_absolute_pose_vs_sparse_proprio`: predicted delta actions
   accumulated with the same global-EE convention as
   `evaluation/X-WAM/deploy_policy.py::compute_future_poses`, then compared to
   the predicted sparse proprio nodes at offsets 0, 4, ..., 32.

The third item is an internal consistency diagnostic between two model outputs;
it is deliberately not a GT metric.

Raw RGB is AV1, so the script extracts selected frames with `ffmpeg` rather
than decord. It requires `pyarrow` in the X-WAM virtual environment (the raw
data converter requires it too).

```bash
.venv/bin/python deployment/evaluate_raw_action_fit.py \
    --raw-dataset raw_data/place_fruits_in_bucket_v1 \
    --exp-path experiments/multitask_merged-v1-sft \
    --wan-checkpoint-dir checkpoints/Wan2.2-TI2V-5B \
    --deployment-checkpoint experiments/multitask_merged-v1-sft/checkpoints/last.deployment.pt \
    --max-queries 24 --frame-stride 96 \
    --output deployment/place_fruits_action_fit.json
```

The JSON has the three reports under `tests`, both in aggregate and per query.
Absolute EE position errors are millimetres; orientation uses sign-invariant
quaternion geodesic degrees; gripper errors are radians. Unless
`--no-save-actions` is passed, each query also retains the raw predictions, GT,
and accumulated absolute-action poses. `first_action` is the most direct delta
teacher-forcing metric; later horizons are an open-loop chunk comparison. By
default it samples valid action starts in the 10%--90% range of each episode.
Use `--min-frame-fraction 0 --max-frame-fraction 1` for the all-episode range,
or set both to `0` for initial-pose-only analysis.

### `get_config` metadata

```json
{
  "server": "xwam-websocket",
  "num_views": 3,
  "view_order": ["fixed_front", "left_arm", "right_arm"],
  "video_size": [256, 320],
  "crop_ratio": 0.95,
  "proprio_dim": 16,
  "action_dim": 14,
  "action_horizon": 32,
  "action_fps": 30.0,
  "proprio_fps": 7.5,
  "sample_steps": 50,
  "action_denoise_steps": 10,
  "action_representation": "delta_ee_global_relative_to_state",
  "tasks": ["...5 task strings..."],
  "prompt_must_be_task": true
}
```

## Client

Any msgpack websockets client works (`deployment/lerobot_client_example.py`);
`openpi_client.WebsocketClientPolicy` is protocol-compatible:

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
policy = WebsocketClientPolicy("10.0.0.5", 8080)
result = policy.infer({"images": rgb, "agent_pos": joints, "prompt": "Fold the towel."})
```

## Conventions & gotchas

- **Video latent sampling is retained at serving time** — video predictions
  are not sent to the client, but their denoising trajectory conditions the
  action/proprio predictions. Depth branches are not constructed.
- Startup uses memory-mapped checkpoint loading and direct BF16 base-model
  construction. This avoids the previous CPU peak where the full 37 GB SFT
  checkpoint and an FP32 base model could coexist in RAM.
- Image preprocessing (identical to `evaluation/policy_server.py`):
  resize → center crop (`crop_ratio`) → resize back, then `/127.5 - 1`
  normalization. Do **not** pre-normalize images.
- `proprios`/`actions` are de/normalized with the training quantile statistics
  (q01/q99) from the experiment `config.yaml`; clients send/receive **raw
  physical units**.
- With pre-encoded prompts the checkpoint's T5 weights are ignored (the
  checkpoint ships a frozen T5 copy; embeddings are precomputed from the base
  `models_t5_umt5-xxl-enc-bf16.pth` — bit-identical unless T5 was fine-tuned).
- Inference is serialized by a lock — one request at a time.

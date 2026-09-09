# X-WAM Real-Robot Deployment

OpenPI-style WebSocket server serving an X-WAM SFT checkpoint for real-robot
control. One binary websocket frame per request; payload/result are python
dicts serialized with msgpack + msgpack-numpy (numpy arrays travel inline).

## Server components

| File | Purpose |
|---|---|
| `websocket_policy_server.py` | WebSocket server (openpi-compatible wire protocol) |
| `xwam_policy.py` | Inference wrapper: preprocessing, checkpoint loading, denormalization |
| `precompute_prompt_embeddings.py` | Offline T5 prompt-embedding pre-encoding (small-GPU deployment) |
| `lerobot_client_example.py` | Minimal client example (no `openpi_client` dependency) |

## What the server loads

- `<exp-path>/config.yaml` — model/data config, quantile statistics (q01/q99),
  `video_size`, `crop_ratio`, `fps`, `frame_skip`, `action_skip`
- `<exp-path>/checkpoints/<steps>.ckpt/checkpoint/mp_rank_00_model_states.pt` —
  DeepSpeed ZeRO model weights (only `module`; depth branch + T5 tensors skipped)
- `wan_checkpoint_dir` (default from config, e.g. `./checkpoints/Wan2.2-TI2V-5B/`) —
  VAE (`Wan2.2_VAE.pth`), DiT architecture + base weights
  (`diffusion_pytorch_model-*.safetensors`)
- `<exp-path>/prompt_embeddings.pt` (optional, auto-detected; override with
  `--prompt-embeddings`) — pre-encoded T5 prompt embeddings. When present, the
  11 GB T5 encoder is **not loaded** and `prompt` must be exactly one of the
  encoded task strings. Peak VRAM ~12 GB (fits a 24 GB RTX 3090).

## Running

```bash
# 1. (one-time) pre-encode the task prompts
.venv/bin/python deployment/precompute_prompt_embeddings.py \
    --tasks-jsonl raw_data/multitask_merged_v1/meta/tasks.jsonl \
    --wan-checkpoint-dir ./checkpoints/Wan2.2-TI2V-5B \
    --config experiments/multitask_merged-v1-sft/config.yaml \
    --dst experiments/multitask_merged-v1-sft

# 2. start the server
.venv/bin/python deployment/websocket_policy_server.py \
    --exp-path experiments/multitask_merged-v1-sft \
    --host 0.0.0.0 --port 8080
```

## Wire protocol

One binary frame per request, msgpack dict. Control commands:

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
| `actions` | `float32 [Ta, 14]` | Delta end-effector actions at `action_fps` Hz: per step `[l_dxyz(3), l_drotvec(3), l_dgrip(1), r_...]`, **relative to the pose at request time** (global frame). Robot side accumulates deltas itself (reference impl: `evaluation/X-WAM/deploy_policy.py::compute_future_poses`). |
| `proprios` | `float32 [Tp, 16]` | Predicted absolute EE chain at `proprio_fps` Hz, same convention as input `proprios`. |
| `action_fps` | `float` | `raw_fps / action_skip` (e.g. 30/1 = 30 Hz with this checkpoint's `action_skip=1`... see `get_config` for the live value). |
| `proprio_fps` | `float` | `raw_fps / frame_skip` (e.g. 30/4 = 7.5 Hz). |
| `view_order` | `list[str]` | The view order the server expects (`["fixed_front", "left_arm", "right_arm"]`). |
| `prompt` / `seed` | `str` / `int` | Echo of the request (effective values). |
| `infer_time_s` | `float` | Server-side wall-clock inference time. |
| `error` | `str` | Present only on failure (bad prompt, bad shapes, ...). |

For `multitask_merged-v1-sft`: `Ta = 32` (i.e. `(frame_num-1) * frame_skip / action_skip`
= 8*4), `Tp = 9`.

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
  "action_fps": 30.0,
  "proprio_fps": 7.5,
  "sample_steps": 50,
  "action_denoise_steps": 10,
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

- **No video/depth prediction at serving time** — only actions/proprios are
  returned (depth branch is not even constructed; video latents participate
  internally as conditions only).
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

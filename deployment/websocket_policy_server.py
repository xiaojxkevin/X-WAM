"""OpenPI-style WebSocket policy server for X-WAM.

Protocol (compatible with ``openpi_client.websocket_client_policy.WebsocketClientPolicy``):
  - Immediately after a WebSocket connection is established, the server sends
    ``{"metadata": ...}``. OpenPI clients wait for this frame before sending
    their first inference request.
  - One binary websocket frame per request.
  - Payload/result are python dicts serialized with msgpack + the bundled
    OpenPI-compatible NumPy codec (numpy arrays travel inline).
  - The first request may be ``{"command": "get_config"}`` to fetch server metadata;
    otherwise the dict is treated as an observation and forwarded to the policy.

Run::

    python deployment/websocket_policy_server.py \
        --exp-path experiments/place_fruits_in_bucket-v1-sft \
        --wan-checkpoint-dir ./checkpoints/Wan2.2-TI2V-5B \
        --host 0.0.0.0 --port 8080

Client (on the robot side)::

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy = WebsocketClientPolicy("10.0.0.5", 8080)
    result = policy.infer({"images": rgb, "proprios": proprio, "prompt": "..."})
"""

import argparse
import asyncio
import logging
import os
import threading
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from websockets.asyncio.server import serve

import openpi_msgpack_numpy
import xwam_policy


def pack(obj: dict) -> bytes:
    # The robot runs OpenPI's vendored msgpack-numpy codec.  Its ndarray wire
    # format differs from the PyPI ``msgpack_numpy`` package installed here.
    return openpi_msgpack_numpy.packb(obj)


def unpack(data: bytes) -> dict:
    return openpi_msgpack_numpy.unpackb(data, raw=False)


class WebSocketPolicyServer:
    def __init__(self, policy: "xwam_policy.XWAMPolicy", host: str = "0.0.0.0", port: int = 8080):
        self._policy = policy
        self._host = host
        self._port = port
        self._lock = threading.Lock()  # one GPU inference at a time

    @property
    def metadata(self) -> dict:
        p = self._policy
        return {
            "protocol_version": 1,
            "server": "xwam-websocket",
            "num_views": 3,
            "view_order": ["fixed_front", "left_arm", "right_arm"],
            "video_size": list(p.video_size),
            "crop_ratio": p.crop_ratio,
            "proprio_dim": 16,
            "action_dim": len(p.action_q01),
            "action_horizon": (int(p.config.frame_num) - 1) * p.frame_skip // p.action_skip,
            "action_fps": p.raw_fps / p.action_skip,
            "proprio_fps": p.raw_fps / p.frame_skip,
            "action_representation": "delta_ee_global_relative_to_state",
            "sample_steps": p.config.sample_steps,
            "action_denoise_steps": p.config.action_denoise_steps,
            "inference_backend": "compile",
            "tasks": p.tasks,
            "prompt_must_be_task": bool(p.prompt_to_embedding),
        }

    async def _handler(self, websocket) -> None:
        peer = websocket.remote_address
        logging.info(f"client connected: {peer}")
        try:
            # OpenPI's WebsocketClientPolicy blocks on recv() immediately
            # after the handshake to obtain this frame. Sending it proactively
            # avoids a client/server deadlock before the first inference.
            await websocket.send(pack({"metadata": self.metadata}))
            async for message in websocket:
                try:
                    payload = unpack(message)
                except Exception as exc:
                    await websocket.send(pack({"error": f"failed to decode payload: {exc}"}))
                    continue

                command = payload.get("command")
                if command == "get_config":
                    await websocket.send(pack({"metadata": self.metadata}))
                    continue
                if command == "ping":
                    await websocket.send(pack({"pong": time.time()}))
                    continue
                payload.pop("_metadata", None)

                # GPU inference is blocking; run it in a thread so websocket pings stay alive.
                result = await asyncio.to_thread(self._infer_locked, payload)
                await websocket.send(pack(result))
        except Exception as exc:
            logging.exception(f"connection {peer} failed: {exc}")
        finally:
            logging.info(f"client disconnected: {peer}")

    def _infer_locked(self, payload: dict) -> dict:
        with self._lock:
            try:
                return self._policy.infer(payload)
            except Exception as exc:
                logging.exception("inference failed")
                return {"error": str(exc)}

    async def serve(self) -> None:
        logging.info(f"serving on ws://{self._host}:{self._port}")
        async with serve(self._handler, self._host, self._port, max_size=64 * 1024 * 1024):
            await asyncio.get_running_loop().create_future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-path", type=str, required=True, help="Experiment dir containing config.yaml + checkpoints/.")
    parser.add_argument("--steps", type=str, default="last", help="Checkpoint step name (dir under checkpoints/).")
    parser.add_argument("--wan-checkpoint-dir", type=str, default=None, help="Wan2.2-TI2V-5B base weights dir.")
    parser.add_argument(
        "--deployment-checkpoint",
        type=str,
        default=None,
        help="Optional inference-only checkpoint generated by export_deployment_checkpoint.py.",
    )
    parser.add_argument("--denoise-steps", type=int, default=50, help="Full video/depth denoising steps.")
    parser.add_argument("--action-denoise-steps", type=int, default=5, help="Async action/proprio denoising steps (0 = joint).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include per-request GPU time and allocated/reserved/peak VRAM in each response and server log.",
    )
    parser.add_argument(
        "--prompt-embeddings",
        type=str,
        default=None,
        help="prompt_embeddings.pt from deployment/precompute_prompt_embeddings.py "
        "(default: <exp-path>/prompt_embeddings.pt if present). Skips loading T5.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = xwam_policy.XWAMPolicy(
        exp_path=args.exp_path,
        steps=args.steps,
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        denoise_steps=args.denoise_steps,
        action_denoise_steps=args.action_denoise_steps,
        profile=args.profile,
        prompt_embeddings=args.prompt_embeddings,
        deployment_checkpoint=args.deployment_checkpoint,
    )
    server = WebSocketPolicyServer(policy, host=args.host, port=args.port)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()

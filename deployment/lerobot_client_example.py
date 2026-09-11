"""Example LeRobot-side client: send an observation, receive an action chunk.

No openpi dependency needed — this reimplements the openpi wire protocol with
plain ``websockets`` + ``msgpack``. If the robot already has ``openpi_client``
installed, ``WebsocketClientPolicy`` works identically.

Protocol:
  - send one binary frame: msgpack(packb, numpy via the bundled OpenPI codec) dict
  - receive one binary frame: msgpack dict
"""

import argparse
import asyncio
import json
import time

import numpy as np
import websockets

import openpi_msgpack_numpy

def pack(obj: dict) -> bytes:
    return openpi_msgpack_numpy.packb(obj)


def unpack(data: bytes) -> dict:
    return openpi_msgpack_numpy.unpackb(data, raw=False)


async def main(args: argparse.Namespace) -> None:
    async with websockets.connect(args.server, max_size=64 * 1024 * 1024) as ws:
        # 1. the server sends OpenPI-compatible metadata immediately after the handshake
        meta = unpack(await ws.recv())
        print("server metadata:", json.dumps(meta, indent=2))

        # 2. Observation loop (zero-valued dummy input; it never commands a robot).
        #    images: 3 views, ALPHABETICAL order = [fixed_front, left_arm, right_arm]
        images = np.stack(
            [np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8) for _ in range(3)]
        )  # V, H, W, 3
        agent_pos = np.zeros(14, dtype=np.float32)  # raw joints [l_j1..6, l_grip, r_j1..6, r_grip]

        payload = {
            "images": images,
            "agent_pos": agent_pos,  # OR "proprios": np.float32[16] raw EE
            "prompt": args.prompt,
            "seed": args.seed,
            # "cfg": 0.0,      # optional CFG scale
        }

        # ``warmup_requests`` are excluded from the reported latency.  Use a
        # Fixed seed so repeated compiled runs are directly comparable.
        wall_times, server_times = [], []
        result = None
        for request_idx in range(args.warmup_requests + args.requests):
            start = time.perf_counter()
            await ws.send(pack(payload))
            current_result = unpack(await ws.recv())
            wall_time = time.perf_counter() - start

            if "error" in current_result:
                raise RuntimeError(current_result["error"])
            result = current_result
            if request_idx >= args.warmup_requests:
                wall_times.append(wall_time)
                server_times.append(float(current_result["infer_time_s"]))

        assert result is not None

        # 3. result: delta EE actions at action_fps (30 Hz when action_skip=1)
        actions = result["actions"]  # [Ta, 14]: [l_dxyz(3), l_drotvec(3), l_dgrip(1), r_...]
        proprios = result["proprios"]  # [Tp, 16] absolute EE chain, raw units
        print(f"actions {actions.shape} @ {result['action_fps']:.1f} Hz, "
              f"proprios {proprios.shape} @ {result['proprio_fps']:.1f} Hz, "
              f"backend={result['inference_backend']}")
        print(
            f"{len(server_times)} measured requests after {args.warmup_requests} client warmups | "
            f"server p50/p95={np.percentile(server_times, [50, 95])} s | "
            f"client p50/p95={np.percentile(wall_times, [50, 95])} s"
        )
        if "profile" in result:
            print("server GPU profile:", json.dumps(result["profile"], indent=2))
        print("first action:", np.round(actions[0], 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8080", help="X-WAM WebSocket endpoint.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Exact task string in prompt_embeddings.pt; inspect the server metadata for valid values.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for comparable output and latency runs.")
    parser.add_argument("--warmup-requests", type=int, default=5, help="Requests excluded from latency statistics.")
    parser.add_argument("--requests", type=int, default=30, help="Measured requests after warmup.")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup_requests < 0:
        parser.error("--requests must be positive and --warmup-requests must be non-negative.")

    asyncio.run(main(args))

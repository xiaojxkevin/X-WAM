"""Example LeRobot-side client: send an observation, receive an action chunk.

No openpi dependency needed — this reimplements the openpi wire protocol with
plain ``websockets`` + ``msgpack``. If the robot already has ``openpi_client``
installed, ``WebsocketClientPolicy`` works identically.

Protocol:
  - send one binary frame: msgpack(packb, numpy via msgpack_numpy) dict
  - receive one binary frame: msgpack dict
"""

import json
import numpy as np
import msgpack
import msgpack_numpy
import websockets

SERVER = "ws://10.0.0.5:8080"  # <-- server host:port


def pack(obj: dict) -> bytes:
    return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


async def main() -> None:
    async with websockets.connect(SERVER, max_size=64 * 1024 * 1024) as ws:
        # 1. optional: fetch server metadata
        await ws.send(pack({"command": "get_config"}))
        meta = unpack(await ws.recv())
        print("server metadata:", json.dumps(meta, indent=2))

        # 2. observation loop (example data below)
        #    images: 3 views, ALPHABETICAL order = [fixed_front, left_arm, right_arm]
        images = np.stack(
            [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        )  # V, H, W, 3
        agent_pos = np.zeros(14, dtype=np.float32)  # raw joints [l_j1..6, l_grip, r_j1..6, r_grip]

        payload = {
            "images": images,
            "agent_pos": agent_pos,  # OR "proprios": np.float32[16] raw EE
            "prompt": "Place both fruits into the bucket.",
            # "seed": 123,     # optional deterministic sampling
            # "cfg": 0.0,      # optional CFG scale
        }
        await ws.send(pack(payload))
        result = unpack(await ws.recv())

        if "error" in result:
            raise RuntimeError(result["error"])

        # 3. result: delta EE actions at action_fps (~7.5 Hz with frame_skip=4)
        actions = result["actions"]  # [Ta, 14]: [l_dxyz(3), l_drotvec(3), l_dgrip(1), r_...]
        proprios = result["proprios"]  # [Tp, 16] absolute EE chain, raw units
        print(f"actions {actions.shape} @ {result['action_fps']:.1f} Hz, "
              f"proprios {proprios.shape} @ {result['proprio_fps']:.1f} Hz, "
              f"inferred in {result['infer_time_s']:.2f}s")
        print("first action:", np.round(actions[0], 4))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

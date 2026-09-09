"""Create a minimal X-WAM inference checkpoint without DeepSpeed training state.

The original ``mp_rank_00_model_states.pt`` is a DeepSpeed resume checkpoint.
Besides the model it contains T5, depth-branch weights, and frozen-parameter
fragments used only to resume training. The real-robot server uses precomputed
T5 embeddings and ``use_depth=False``, so it needs only the shared
DiT/action/proprio weights and the VAE.

This script intentionally never builds ``XWAMRunner`` and never touches CUDA.
It memory-maps the source checkpoint and writes tensor references directly to a
new PyTorch archive; do not add tensor ``clone()``, ``cpu()``, or dtype casts
here, as they would create an avoidable RAM peak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


EXCLUDED_MODEL_PREFIXES = ("model.extra_blocks.", "model.extra_heads.")
KEPT_PREFIXES = ("model.", "vae.")


def keep_key(key: str) -> bool:
    """Whether a ``module`` state-dict key is required by no-depth serving."""
    return key.startswith(KEPT_PREFIXES) and not key.startswith(EXCLUDED_MODEL_PREFIXES)


def export_checkpoint(source: Path, destination: Path) -> tuple[int, int]:
    """Write the inference-only state dict and return (tensor_count, bytes)."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if "module" not in checkpoint or not isinstance(checkpoint["module"], dict):
        raise ValueError(f"{source} is not a DeepSpeed model-state checkpoint with a 'module' state dict.")

    module = checkpoint["module"]
    deployment_module = {
        key: tensor
        for key, tensor in module.items()
        if isinstance(tensor, torch.Tensor) and keep_key(key)
    }
    if not deployment_module:
        raise ValueError("No deployment tensors selected; check the source checkpoint format.")

    byte_count = sum(tensor.numel() * tensor.element_size() for tensor in deployment_module.values())
    payload = {
        "format": "xwam-deployment-state-v1",
        "source_checkpoint": str(source),
        "module": deployment_module,
    }
    torch.save(payload, destination)
    return len(deployment_module), byte_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="DeepSpeed mp_rank_00_model_states.pt source.")
    parser.add_argument("--destination", type=Path, required=True, help="New inference-only .pt checkpoint.")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if destination.exists():
        parser.error(f"destination already exists (refusing to overwrite): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensor_count, byte_count = export_checkpoint(source, destination)
    print(
        f"Wrote {destination} ({destination.stat().st_size / 2**30:.2f} GiB on disk; "
        f"{tensor_count} tensors / {byte_count / 2**30:.2f} GiB payload)."
    )


if __name__ == "__main__":
    main()

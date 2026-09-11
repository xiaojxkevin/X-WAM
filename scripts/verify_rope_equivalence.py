#!/usr/bin/env python3
"""Verify a real-valued RoPE implementation against X-WAM's complex reference.

This is deliberately standalone: it does not change ``modules/wan_model.py``.
It compares both forward values and input gradients for the exact RoPE layout
used by X-WAM: ``x=[B, L, H, D]`` and complex frequencies ``[L, D/2]``.

Examples:

    .venv/bin/python scripts/verify_rope_equivalence.py --device cpu
    .venv/bin/python scripts/verify_rope_equivalence.py --device cuda --dtype bfloat16
"""

import argparse

import torch


def rope_params_complex_reference(
    max_seq_len: int, dim: int, theta: float = 10000.0, scale: float = 1.0
) -> torch.Tensor:
    """Exact current ``modules.wan_model.rope_params`` mathematical reference."""
    assert dim % 2 == 0
    phase = torch.outer(
        torch.arange(0, max_seq_len, 1 / scale),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(phase), phase)


def rope_apply_1d_complex_reference(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Exact current ``modules.wan_model.rope_apply_1d`` mathematical reference."""
    b, length, heads, dim = x.shape
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(b, length, heads, dim // 2, 2))
    return torch.view_as_real(x_complex * freqs).flatten(3).float()


def rope_params_real(max_seq_len: int, dim: int, theta: float = 10000.0, scale: float = 1.0) -> torch.Tensor:
    """Candidate interleaved real representation: ``[cos, sin, ...]``.

    Its phase construction intentionally matches :func:`modules.wan_model.rope_params`.
    """
    assert dim % 2 == 0
    phase = torch.outer(
        torch.arange(0, max_seq_len, 1 / scale),
        1.0 / torch.pow(theta, torch.arange(dim // 2, dtype=torch.float64).div(dim // 2)),
    )
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1).flatten(-2)


def rope_apply_1d_real(x: torch.Tensor, freqs_real: torch.Tensor) -> torch.Tensor:
    """Real-arithmetic candidate for ``rope_apply_1d`` with identical pair layout."""
    b, length, heads, dim = x.shape
    pairs = x.to(torch.float64).reshape(b, length, heads, dim // 2, 2)
    freq_pairs = freqs_real.to(torch.float64).reshape(length, dim // 2, 2)
    cos = freq_pairs[..., 0].unsqueeze(0).unsqueeze(2)
    sin = freq_pairs[..., 1].unsqueeze(0).unsqueeze(2)
    real = pairs[..., 0] * cos - pairs[..., 1] * sin
    imag = pairs[..., 0] * sin + pairs[..., 1] * cos
    return torch.stack((real, imag), dim=-1).flatten(3).float()


def max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.detach().abs().max().clamp_min(torch.finfo(expected.dtype).tiny)
    return float((actual - expected).detach().abs().max() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=257)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.head_dim % 2:
        parser.error("--head-dim must be even")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Construct frequency representations independently, then require their
    # cos/sin values to agree before testing the application itself.
    freqs_complex = rope_params_complex_reference(
        args.length, args.head_dim, theta=args.theta, scale=args.scale
    ).to(device)
    freqs_real = rope_params_real(args.length, args.head_dim, theta=args.theta, scale=args.scale).to(device)
    freqs_real_pairs = freqs_real.reshape(args.length, args.head_dim // 2, 2)
    torch.testing.assert_close(freqs_real_pairs[..., 0], freqs_complex.real, rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(freqs_real_pairs[..., 1], freqs_complex.imag, rtol=args.rtol, atol=args.atol)

    shape = (args.batch, args.length, args.heads, args.head_dim)
    x_reference = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    x_candidate = x_reference.detach().clone().requires_grad_(True)
    upstream_grad = torch.randn(shape, device=device, dtype=torch.float32)

    reference = rope_apply_1d_complex_reference(x_reference, freqs_complex)
    candidate = rope_apply_1d_real(x_candidate, freqs_real)
    reference.backward(upstream_grad)
    candidate.backward(upstream_grad)

    torch.testing.assert_close(candidate, reference, rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(x_candidate.grad, x_reference.grad, rtol=args.rtol, atol=args.atol)

    print(
        "PASS"
        f" device={device} dtype={dtype} shape={shape}"
        f" forward_max_abs={(candidate - reference).abs().max().item():.3e}"
        f" forward_max_rel={max_relative_error(candidate, reference):.3e}"
        f" grad_max_abs={(x_candidate.grad - x_reference.grad).abs().max().item():.3e}"
        f" grad_max_rel={max_relative_error(x_candidate.grad, x_reference.grad):.3e}"
    )


if __name__ == "__main__":
    main()

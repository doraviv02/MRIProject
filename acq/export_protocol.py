"""Protocol exporter: converts learned w* into a human-readable acquisition protocol.

Produces:
  - A JSON/CSV file listing (phase_encode_line, averages) for each line.
  - A mask image (w_m > 0 = acquired).
  - The real-reps realization: which M4Raw repetitions to use per line.
"""

import json
import os
import numpy as np
import torch
from typing import List, Optional


def export_protocol(
    w_int: torch.Tensor,
    out_dir: str = "results/",
    name: str = "learned_protocol",
) -> dict:
    """
    Export the integer allocation as a human-readable protocol.

    Args:
        w_int:   [N] integer allocation (LongTensor or np.ndarray)
        out_dir: output directory
        name:    protocol name prefix

    Returns:
        protocol dict
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(w_int, torch.Tensor):
        w_int = w_int.cpu().numpy()

    N = len(w_int)
    acquired_lines = [int(m) for m in range(N) if w_int[m] > 0]
    n_acquired = len(acquired_lines)
    n_unacquired = N - n_acquired
    total_budget = int(w_int.sum())

    protocol = {
        "n_lines": N,
        "budget": total_budget,
        "n_acquired_lines": n_acquired,
        "n_unacquired_lines": n_unacquired,
        "acceleration": N / max(n_acquired, 1),
        "max_averages": int(w_int.max()),
        "mean_averages_acquired": float(w_int[w_int > 0].mean()) if n_acquired > 0 else 0.0,
        "line_allocations": [
            {"line": int(m), "averages": int(w_int[m])}
            for m in range(N)
        ],
    }

    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Protocol saved to {json_path}")

    # Also save as CSV
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w") as f:
        f.write("phase_encode_line,averages\n")
        for m in range(N):
            f.write(f"{m},{w_int[m]}\n")
    print(f"CSV saved to {csv_path}")

    # Save mask array
    mask_path = os.path.join(out_dir, f"{name}_mask.npy")
    np.save(mask_path, (w_int > 0).astype(np.int32))

    return protocol


def realize_reps(
    w_int: torch.Tensor,
    kspace_reps: torch.Tensor,
    max_reps: int = 3,
) -> torch.Tensor:
    """
    Physically realize an integer allocation by averaging actual M4Raw repetitions.

    Args:
        w_int:       [N] integer allocation, must have w_int[m] ≤ max_reps for all m.
        kspace_reps: [reps, H, W] complex k-space tensor from M4Raw (single-coil).
        max_reps:    Maximum available repetitions.

    Returns:
        y_realized: [H, W] complex k-space; averaged per the protocol, zeros for unacquired lines.
    """
    if isinstance(w_int, torch.Tensor):
        w_arr = w_int.cpu().numpy()
    else:
        w_arr = np.array(w_int)

    if isinstance(kspace_reps, torch.Tensor):
        ks = kspace_reps.cpu().numpy()  # [reps, H, W] complex
    else:
        ks = np.array(kspace_reps)

    reps, H, W = ks.shape
    N = W  # phase-encode lines

    # Clamp allocations to available reps
    w_clamped = np.clip(w_arr, 0, min(max_reps, reps))

    y = np.zeros((H, W), dtype=np.complex64)
    for m in range(N):
        n_avg = int(w_clamped[m])
        if n_avg == 0:
            continue
        # Average the first n_avg repetitions of line m
        y[:, m] = ks[:n_avg, :, m].mean(axis=0)

    return torch.from_numpy(y)


def summarise_policy(w: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> str:
    """Print a human-readable summary of the learned allocation."""
    if isinstance(w, torch.Tensor):
        w_np = w.detach().cpu().numpy()
    else:
        w_np = w
    lines = [
        f"Allocation summary (N={len(w_np)}, B={w_np.sum():.1f})",
        f"  Max   : {w_np.max():.3f}",
        f"  Min   : {w_np.min():.3f}",
        f"  Mean  : {w_np.mean():.3f}",
        f"  Lines with w>1 : {(w_np > 1).sum()}",
        f"  Lines with w<0.1: {(w_np < 0.1).sum()} (effectively unacquired)",
    ]
    if sigma is not None:
        s_np = sigma.cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma
        rho = w_np / (s_np ** 2 + 1e-12)
        lines.append(f"  Precision ρ: min={rho.min():.2f}, max={rho.max():.2f}")
    return "\n".join(lines)

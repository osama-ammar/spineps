"""
Lightweight GPU memory profiler for SPINEPS.
Enable with:  SPINEPS_PROFILE_GPU=1
Reports: allocated, reserved, and peak (max) memory in MiB.
Peak stats are reset at the start of each scan so inference stages show their own peak.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch


def _is_enabled() -> bool:
    return os.environ.get("SPINEPS_PROFILE_GPU", "").strip().lower() in ("1", "true", "yes")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def get_gpu_memory_mb() -> dict[str, float] | None:
    """Current and peak GPU memory in MiB. Returns None if CUDA not available or profiling disabled."""
    if not _cuda_available():
        return None
    torch.cuda.synchronize()
    # 1 byte = 1 / (1024**2) MiB
    scale = 1.0 / (1024.0 * 1024.0)
    out: dict[str, float] = {
        "allocated_mb": torch.cuda.memory_allocated() * scale,
        "reserved_mb": torch.cuda.memory_reserved() * scale,
        "max_allocated_mb": torch.cuda.max_memory_allocated() * scale,
        "max_reserved_mb": torch.cuda.max_memory_reserved() * scale,
    }
    try:
        # PyTorch 1.12+
        out["allocated_peak_mb"] = torch.cuda.memory_allocated() * scale  # current; peak is max_allocated
    except Exception:
        pass
    return out


def format_memory_line(stats: dict[str, float]) -> str:
    return (
        f"allocated={stats['allocated_mb']:.1f} MiB  reserved={stats['reserved_mb']:.1f} MiB  "
        f"peak_alloc={stats['max_allocated_mb']:.1f} MiB  peak_reserved={stats['max_reserved_mb']:.1f} MiB"
    )


def log_gpu_memory(label: str, file: Any = None) -> dict[str, float] | None:
    """
    If SPINEPS_PROFILE_GPU=1 and CUDA is available, log current/peak GPU memory and return stats.
    Otherwise no-op and return None.
    """
    if not _is_enabled():
        return None
    stats = get_gpu_memory_mb()
    if stats is None:
        return None
    line = f"[GPU] {label}  {format_memory_line(stats)}\n"
    if file is not None:
        try:
            file.write(line)
            file.flush()
        except Exception:
            pass
    # Always print so user sees it in terminal
    sys.stdout.write(line)
    sys.stdout.flush()
    return stats


def reset_peak_stats() -> None:
    """Reset PyTorch peak memory counters. Call after model load to measure inference peak only."""
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def empty_cache() -> None:
    """Release unused cached GPU memory."""
    if _cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

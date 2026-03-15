"""Configuration loading for ThinkBudget."""

from __future__ import annotations

import json
import os
from pathlib import Path

from thinkbudget.models import BudgetConfig, ComplexityTier, ProxyConfig


def load_config(path: str | Path | None = None) -> ProxyConfig:
    """Load proxy configuration from file or environment."""
    config = ProxyConfig()

    # Override from config file
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        config = ProxyConfig(**data)

    # Override from environment
    if url := os.getenv("THINKBUDGET_BACKEND_URL"):
        config.backend_url = url
    if model := os.getenv("THINKBUDGET_MODEL"):
        config.backend_model = model
    if port := os.getenv("THINKBUDGET_PORT"):
        config.port = int(port)
    if gpu_cost := os.getenv("THINKBUDGET_GPU_COST_PER_HOUR"):
        config.budget.gpu_cost_per_hour = float(gpu_cost)
    if gpu_name := os.getenv("THINKBUDGET_GPU_NAME"):
        config.budget.gpu_name = gpu_name

    return config


# GPU pricing for common CloudRift GPUs
GPU_PRICING: dict[str, float] = {
    "RTX 4090": 0.39,
    "RTX 5090": 0.65,
    "RTX PRO 6000": 1.29,
    "L40S": 0.63,
    "H100": 1.70,
    "H200": 2.50,
    "A100": 1.50,
    "A10G": 0.50,
}


def get_gpu_cost(gpu_name: str) -> float:
    """Get hourly cost for a GPU model."""
    for name, cost in GPU_PRICING.items():
        if name.lower() in gpu_name.lower():
            return cost
    return 0.39  # Default to RTX 4090

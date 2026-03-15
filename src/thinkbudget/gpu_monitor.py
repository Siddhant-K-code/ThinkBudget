"""GPU cost monitor for ThinkBudget.

Tracks real-time GPU metrics (power, utilization, memory) and computes
per-query energy cost in joules and dollars. Falls back to estimation
when pynvml is unavailable (e.g., no GPU present).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from thinkbudget.models import QueryCost

# Try to import pynvml for real GPU monitoring
_HAS_NVML = False
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    pass


@dataclass
class GPUSnapshot:
    """Point-in-time GPU metrics."""

    timestamp: float
    power_watts: float
    gpu_utilization: float  # 0-100
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float


@dataclass
class GPUInfo:
    """Static GPU information."""

    name: str = "Unknown"
    memory_total_mb: float = 0
    driver_version: str = ""
    cuda_version: str = ""
    tdp_watts: float = 350  # Default TDP


class GPUMonitor:
    """Monitors GPU metrics and computes per-query costs.

    Uses pynvml when available, falls back to estimation based on
    known GPU specs and token throughput.
    """

    def __init__(self, gpu_cost_per_hour: float = 0.39, device_index: int = 0):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.device_index = device_index
        self._handle = None
        self._initialized = False
        self._gpu_info = GPUInfo()
        self._sampling = False
        self._samples: list[GPUSnapshot] = []
        self._sample_thread: threading.Thread | None = None

        self._try_init()

    def _try_init(self):
        """Try to initialize NVML."""
        if not _HAS_NVML:
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)

            self._gpu_info = GPUInfo(
                name=name,
                memory_total_mb=mem_info.total / (1024 * 1024),
                driver_version=pynvml.nvmlSystemGetDriverVersion(),
                tdp_watts=power_limit / 1000,
            )
            self._initialized = True
        except Exception:
            self._initialized = False

    @property
    def is_available(self) -> bool:
        return self._initialized

    @property
    def gpu_info(self) -> GPUInfo:
        return self._gpu_info

    def snapshot(self) -> GPUSnapshot:
        """Take a single GPU metrics snapshot."""
        if not self._initialized or self._handle is None:
            return GPUSnapshot(
                timestamp=time.time(),
                power_watts=0,
                gpu_utilization=0,
                memory_used_mb=0,
                memory_total_mb=0,
                temperature_c=0,
            )

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000  # mW -> W
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )

            return GPUSnapshot(
                timestamp=time.time(),
                power_watts=power,
                gpu_utilization=util.gpu,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
                temperature_c=temp,
            )
        except Exception:
            return GPUSnapshot(
                timestamp=time.time(),
                power_watts=0,
                gpu_utilization=0,
                memory_used_mb=0,
                memory_total_mb=0,
                temperature_c=0,
            )

    def start_sampling(self, interval_ms: int = 50):
        """Start background GPU sampling for a query."""
        self._samples = []
        self._sampling = True

        def _sample_loop():
            while self._sampling:
                self._samples.append(self.snapshot())
                time.sleep(interval_ms / 1000)

        self._sample_thread = threading.Thread(target=_sample_loop, daemon=True)
        self._sample_thread.start()

    def stop_sampling(self) -> list[GPUSnapshot]:
        """Stop sampling and return collected snapshots."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None
        samples = list(self._samples)
        self._samples = []
        return samples

    def compute_cost(
        self,
        samples: list[GPUSnapshot] | None = None,
        duration_seconds: float = 0,
        thinking_tokens: int = 0,
    ) -> QueryCost:
        """Compute the cost of a query from GPU samples or estimation.

        If real GPU samples are available, uses actual power measurements.
        Otherwise, estimates based on token count and known GPU specs.
        """
        if samples and len(samples) >= 2:
            return self._cost_from_samples(samples)
        else:
            return self._estimate_cost(duration_seconds, thinking_tokens)

    def _cost_from_samples(self, samples: list[GPUSnapshot]) -> QueryCost:
        """Compute cost from actual GPU power samples."""
        if len(samples) < 2:
            return QueryCost()

        duration = samples[-1].timestamp - samples[0].timestamp
        if duration <= 0:
            return QueryCost()

        # Compute energy via trapezoidal integration of power over time
        energy_joules = 0.0
        for i in range(1, len(samples)):
            dt = samples[i].timestamp - samples[i - 1].timestamp
            avg_power = (samples[i].power_watts + samples[i - 1].power_watts) / 2
            energy_joules += avg_power * dt

        avg_power = sum(s.power_watts for s in samples) / len(samples)
        avg_util = sum(s.gpu_utilization for s in samples) / len(samples)
        avg_mem = sum(s.memory_used_mb for s in samples) / len(samples)

        # Dollar cost = (duration / 3600) * hourly_rate
        dollar_cost = (duration / 3600) * self.gpu_cost_per_hour

        return QueryCost(
            energy_joules=round(energy_joules, 4),
            power_watts_avg=round(avg_power, 2),
            duration_seconds=round(duration, 4),
            dollar_cost=round(dollar_cost, 8),
            gpu_utilization_avg=round(avg_util, 2),
            memory_used_mb=round(avg_mem, 2),
        )

    def _estimate_cost(
        self, duration_seconds: float, thinking_tokens: int
    ) -> QueryCost:
        """Estimate cost when no GPU samples are available.

        Uses a simple model: cost = duration * hourly_rate / 3600
        Power is estimated from GPU TDP and assumed utilization.
        """
        if duration_seconds <= 0:
            # Estimate duration from token count
            # Typical: ~30-80 tokens/sec on RTX 4090 for reasoning models
            tokens_per_sec = 50
            duration_seconds = thinking_tokens / tokens_per_sec

        # Assume 70% of TDP during inference
        estimated_power = self._gpu_info.tdp_watts * 0.7
        energy_joules = estimated_power * duration_seconds
        dollar_cost = (duration_seconds / 3600) * self.gpu_cost_per_hour

        return QueryCost(
            energy_joules=round(energy_joules, 4),
            power_watts_avg=round(estimated_power, 2),
            duration_seconds=round(duration_seconds, 4),
            dollar_cost=round(dollar_cost, 8),
            gpu_utilization_avg=70.0,
            memory_used_mb=0,
        )

    def shutdown(self):
        """Clean up NVML."""
        self._sampling = False
        if self._initialized and _HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

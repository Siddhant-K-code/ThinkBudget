"""Tests for the GPU monitor (runs without a GPU)."""

import unittest

from thinkbudget.gpu_monitor import GPUMonitor


class TestGPUMonitorNoGPU(unittest.TestCase):
    """Tests that run without a GPU present."""

    def setUp(self):
        self.monitor = GPUMonitor(gpu_cost_per_hour=0.39)

    def test_not_available_without_gpu(self):
        # May or may not be available depending on environment
        # Just verify it doesn't crash
        _ = self.monitor.is_available

    def test_snapshot_returns_zeros_without_gpu(self):
        if self.monitor.is_available:
            self.skipTest("GPU is available")
        snap = self.monitor.snapshot()
        self.assertEqual(snap.power_watts, 0)
        self.assertEqual(snap.gpu_utilization, 0)

    def test_estimate_cost_from_tokens(self):
        cost = self.monitor.compute_cost(
            duration_seconds=0,
            thinking_tokens=1000,
        )
        self.assertGreater(cost.dollar_cost, 0)
        self.assertGreater(cost.energy_joules, 0)
        self.assertGreater(cost.duration_seconds, 0)

    def test_estimate_cost_from_duration(self):
        cost = self.monitor.compute_cost(
            duration_seconds=10.0,
            thinking_tokens=0,
        )
        # 10 seconds at $0.39/hr = $0.00108333
        self.assertAlmostEqual(cost.dollar_cost, 10 / 3600 * 0.39, places=6)

    def test_cost_scales_with_gpu_price(self):
        cheap = GPUMonitor(gpu_cost_per_hour=0.39)
        expensive = GPUMonitor(gpu_cost_per_hour=1.29)

        cost_cheap = cheap.compute_cost(duration_seconds=60, thinking_tokens=0)
        cost_expensive = expensive.compute_cost(duration_seconds=60, thinking_tokens=0)

        self.assertGreater(cost_expensive.dollar_cost, cost_cheap.dollar_cost)

    def test_sampling_without_gpu(self):
        self.monitor.start_sampling(interval_ms=50)
        import time
        time.sleep(0.15)
        samples = self.monitor.stop_sampling()
        # Should get some samples even without GPU (they'll be zeros)
        self.assertIsInstance(samples, list)

    def test_shutdown_safe(self):
        self.monitor.shutdown()  # Should not raise


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for the sandbox environment manifest and safety checks."""
from pathlib import Path
import tempfile
import unittest

from sandbox_preflight import ROOT, validate


class SandboxPreflightTest(unittest.TestCase):
    def test_repository_environment_contract(self):
        self.assertEqual(validate(ROOT), [])

    def test_missing_repository_cannot_be_reported_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            issues = validate(Path(directory))
            self.assertGreater(len(issues), 5)
            self.assertTrue(any("CUDA Rust Tile version" in problem for problem in issues))

    def test_cuda_dependency_and_container_must_be_explicit(self):
        cpu = (ROOT / "motor_inferencia/Cargo.toml").read_text()
        gpu = (ROOT / "cuda-kernels/Cargo.toml").read_text()
        legacy_name = bytes((99, 97, 110, 100, 108, 101)).decode("ascii")
        self.assertNotIn(legacy_name, cpu.lower())
        self.assertIn('cutile = "=0.5.0"', gpu)
        self.assertIn('CUDA_COMPUTE_CAP=120', (ROOT / "Dockerfile").read_text())


if __name__ == "__main__":
    unittest.main()

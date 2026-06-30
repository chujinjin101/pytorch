# Owner(s): ["module: tests"]

import os
import pickle
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAcceleratorMemory(TestCase):
    def test_snapshot_dispatches_to_backend_private_api(self):
        class BackendMemory:
            @staticmethod
            def _snapshot(device=None, augment_with_fx_traces=False):
                return {
                    "device": device,
                    "augment_with_fx_traces": augment_with_fx_traces,
                }

        acc = SimpleNamespace(type="foo")
        mod = SimpleNamespace(memory=BackendMemory)
        with (
            patch("torch.accelerator.current_accelerator", return_value=acc),
            patch("torch.get_device_module", return_value=mod) as get_device_module,
        ):
            self.assertEqual(
                torch.accelerator.memory.snapshot(
                    "foo:0", augment_with_fx_traces=True
                ),
                {"device": "foo:0", "augment_with_fx_traces": True},
            )
            get_device_module.assert_called_once_with(acc)

    def test_record_memory_history_dispatches_to_backend_public_api(self):
        calls = []

        class BackendMemory:
            @staticmethod
            def record_memory_history(*args, **kwargs):
                calls.append((args, kwargs))

        acc = SimpleNamespace(type="foo")
        mod = SimpleNamespace(memory=BackendMemory)
        with (
            patch("torch.accelerator.current_accelerator", return_value=acc),
            patch("torch.get_device_module", return_value=mod),
        ):
            torch.accelerator.memory.record_memory_history(
                "all", context="state", max_entries=8
            )

        self.assertEqual(calls, [(("all",), {"context": "state", "max_entries": 8})])

    def test_missing_backend_memory_raises_not_implemented(self):
        acc = SimpleNamespace(type="foo")
        mod = SimpleNamespace()
        with (
            patch("torch.accelerator.current_accelerator", return_value=acc),
            patch("torch.get_device_module", return_value=mod),
            self.assertRaisesRegex(
                NotImplementedError,
                "memory snapshot is not implemented for backend 'foo'",
            ),
        ):
            torch.accelerator.memory.snapshot()

    def test_dump_snapshot_uses_accelerator_snapshot(self):
        snapshot = {"segments": [], "device_traces": []}
        with (
            patch("torch.accelerator.memory.snapshot", return_value=snapshot),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            filename = os.path.join(tmpdir, "snapshot.pickle")
            torch.accelerator.memory.dump_snapshot(filename)
            with open(filename, "rb") as f:
                self.assertEqual(pickle.load(f), snapshot)

    def test_save_usage_uses_accelerator_snapshot(self):
        snapshot = {"segments": [], "device_traces": []}
        with (
            patch("torch.accelerator.memory.snapshot", return_value=snapshot),
            patch("torch.accelerator.memory._segments", return_value="<svg />"),
            patch("torch.accelerator.memory._memory", return_value="<svg />"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            segments_filename = os.path.join(tmpdir, "segments.svg")
            memory_filename = os.path.join(tmpdir, "memory.svg")
            torch.accelerator.memory.save_segment_usage(segments_filename)
            torch.accelerator.memory.save_memory_usage(memory_filename)

            with open(segments_filename) as f:
                self.assertEqual(f.read(), "<svg />")
            with open(memory_filename) as f:
                self.assertEqual(f.read(), "<svg />")


if __name__ == "__main__":
    run_tests()

# Owner(s): ["module: accelerator"]

import torch
import torch.accelerator._sanitizer as asan
import torch.cuda._sanitizer as csan
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAcceleratorSanitizer(TestCase):
    def test_argument_handler_filters_by_device_type(self):
        add_func = torch.ops.aten.add.Tensor
        cpu_tensor = torch.ones(2)

        argument_handler = asan.ArgumentHandler("cuda")
        argument_handler.parse_inputs(
            add_func._schema, (cpu_tensor, cpu_tensor), {}, is_factory=False
        )
        out = torch.add(cpu_tensor, cpu_tensor)
        argument_handler.parse_outputs(add_func._schema, out, is_factory=False)

        self.assertEqual(argument_handler.dataptrs_read, set())
        self.assertEqual(argument_handler.dataptrs_written, set())

    def test_argument_handler_accepts_explicit_device_type(self):
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(2)
        b = torch.zeros(2)

        argument_handler = asan.ArgumentHandler("cpu")
        argument_handler.parse_inputs(add_func._schema, (a, b), {}, is_factory=False)
        out = torch.add(a, b)
        argument_handler.parse_outputs(add_func._schema, out, is_factory=False)

        self.assertEqual(argument_handler.dataptrs_read, {a.data_ptr(), b.data_ptr()})
        self.assertEqual(argument_handler.dataptrs_written, {out.data_ptr()})

    def test_cuda_argument_handler_defaults_to_cuda(self):
        argument_handler = csan.ArgumentHandler()

        self.assertEqual(argument_handler.device_type, "cuda")

    def test_cuda_sanitizer_uses_accelerator_base(self):
        self.assertTrue(issubclass(csan.CUDASanitizer, asan.AcceleratorSanitizer))
        self.assertIs(csan.CUDASanitizerErrors, asan.AcceleratorSanitizerErrors)

    def test_unsupported_backend_error(self):
        sanitizer = asan.AcceleratorSanitizer("cpu")

        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            sanitizer.enable()


if __name__ == "__main__":
    run_tests()

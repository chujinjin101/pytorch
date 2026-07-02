# mypy: allow-untyped-defs
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

from torch.accelerator._sanitizer import (  # noqa: F401
    Access,
    AccessType,
    AcceleratorSanitizer,
    AcceleratorSanitizerDispatchMode,
    AcceleratorSanitizerErrors,
    ArgumentHandler as _ArgumentHandler,
    DEFAULT_STREAM_ID,
    DataPtr,
    EventHandler,
    EventId,
    FACTORY_FUNCTION_REGEX,
    SeqNum,
    StreamId,
    StreamSynchronizations,
    SynchronizationError,
    TensorInfo,
    UnsynchronizedAccessError,
    _TensorsAccessed,
    zip_arguments,
    zip_by_key,
)


class ArgumentHandler(_ArgumentHandler):
    def __init__(self) -> None:
        super().__init__("cuda")


class CUDASanitizerDispatchMode(AcceleratorSanitizerDispatchMode):
    def __init__(self) -> None:
        super().__init__("cuda")


class CUDASanitizer(AcceleratorSanitizer):
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self) -> None:
        super().__init__("cuda")
        self.dispatch = CUDASanitizerDispatchMode()

    def _get_dispatch(self):
        if self.dispatch is None:
            self.dispatch = CUDASanitizerDispatchMode()
        return self.dispatch


CUDASanitizerErrors = AcceleratorSanitizerErrors


def enable_cuda_sanitizer():
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


cuda_sanitizer = CUDASanitizer()


__all__ = [
    "Access",
    "AccessType",
    "ArgumentHandler",
    "CUDASanitizer",
    "CUDASanitizerDispatchMode",
    "CUDASanitizerErrors",
    "DEFAULT_STREAM_ID",
    "DataPtr",
    "EventHandler",
    "EventId",
    "FACTORY_FUNCTION_REGEX",
    "SeqNum",
    "StreamId",
    "StreamSynchronizations",
    "SynchronizationError",
    "TensorInfo",
    "UnsynchronizedAccessError",
    "cuda_sanitizer",
    "enable_cuda_sanitizer",
    "zip_arguments",
    "zip_by_key",
]

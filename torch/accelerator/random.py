from collections.abc import Iterable

import torch
from torch import Tensor

from ._utils import _device_t, _get_device_index, _lazy_call


_NO_ACCELERATOR_ERROR = "No accelerator backend is currently available."


def _get_accelerator() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError(_NO_ACCELERATOR_ERROR)
    return accelerator


def _get_default_generator(device: _device_t = None) -> torch._C.Generator:
    _get_accelerator()
    device_index = _get_device_index(device, optional=True)
    return torch._C._accelerator_getDefaultGenerator(device_index)


def _get_device_index_or_none(device: _device_t = None) -> int | None:
    accelerator = _get_accelerator()
    if device is None:
        return None
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if accelerator.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {accelerator}."
            )
        return device.index
    raise ValueError(f"Expected a torch.device, str, int, or None, but got: {device}")


def _get_current_or_indexed_generator(device_index: int | None) -> torch._C.Generator:
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return torch._C._accelerator_getDefaultGenerator(device_index)


def _is_accelerator_initialized() -> bool:
    accelerator = _get_accelerator()
    device_module = torch.get_device_module(accelerator)
    is_initialized = getattr(device_module, "is_initialized", None)
    return is_initialized() if is_initialized is not None else True


def initial_seed(device: _device_t = None) -> int:
    r"""Return the initial seed of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    on the specified device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the initial seed of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        int: the initial seed of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return _get_default_generator(device).initial_seed()


def get_rng_state(device: _device_t = None) -> Tensor:
    r"""Return the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    as a `torch.Tensor` of dtype `torch.uint8` for the specified accelerator device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Tensor: the RNG state of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return _get_default_generator(device).get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of `torch.Tensor` of dtype `torch.uint8` representing the RNG states of all devices for
    the current :ref:`accelerator<accelerators>`.

    Returns:
        list[torch.Tensor]: the RNG states of the default generators for all devices.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    _get_accelerator()
    return [get_rng_state(i) for i in range(torch.accelerator.device_count())]


def set_rng_state(new_state: Tensor, device: _device_t = None) -> None:
    r"""Set the RNG state of the default :class:`torch.Generator` for the current
    :ref:`accelerator<accelerators>` on the specified device.

    Args:
        new_state (torch.ByteTensor): The desired state.
        device (:class:`torch.device`, str, int, optional): The device to set the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.
    """
    device_index = _get_device_index_or_none(device)
    if not _is_accelerator_initialized():
        with torch._C._DisableFuncTorch():
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    def cb() -> None:
        default_generator = _get_current_or_indexed_generator(device_index)
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the RNG states of all devices for the current :ref:`accelerator<accelerators>`.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    _get_accelerator()
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current
    :ref:`accelerator<accelerators>` device.

    Args:
        seed (int): The desired seed.
    """
    _get_accelerator()
    seed = int(seed)

    def cb() -> None:
        default_generator = _get_current_or_indexed_generator(None)
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all current
    :ref:`accelerator<accelerators>` devices.

    Args:
        seed (int): The desired seed.
    """
    _get_accelerator()
    seed = int(seed)

    def cb() -> None:
        for i in range(torch.accelerator.device_count()):
            default_generator = torch._C._accelerator_getDefaultGenerator(i)
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the
    current :ref:`accelerator<accelerators>` device.
    """
    _get_accelerator()

    def cb() -> None:
        default_generator = _get_current_or_indexed_generator(None)
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all
    current :ref:`accelerator<accelerators>` devices.
    """
    _get_accelerator()

    def cb() -> None:
        random_seed = 0
        seeded = False
        for i in range(torch.accelerator.device_count()):
            default_generator = torch._C._accelerator_getDefaultGenerator(i)
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


__all__ = [
    "initial_seed",
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
]

#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.nn.module import convert_to_npu_module
from intel_npu_acceleration_library.backend.tensor import RemoteTensor
from torch.overrides import TorchFunctionMode
from functools import lru_cache
from typing import Any, MutableMapping
import torch


class NPUDevice(TorchFunctionMode):
    """
    Represents an NPU device.

    This class extends the `TorchFunctionMode` class and provides an implementation
    for the `__torch_function__` method.

    Attributes:
        IMPLEMENTATIONS (MutableMapping[Any, Any]): A dictionary mapping functions to their implementations.

    Methods:
        __torch_function__(func, types, args=(), kwargs=None): Overrides the `__torch_function__`
            method to provide custom behavior for torch functions.

    """

    IMPLEMENTATIONS: MutableMapping[Any, Any] = {}

    def __torch_function__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ):
        """
        Override the torch function behavior for the device class.

        Args:
            func (Any): The torch function being called.
            types (Any): The types of the arguments being passed to the function.
            args (Any, optional): The positional arguments being passed to the function. Defaults to ().
            kwargs (Any, optional): The keyword arguments being passed to the function. Defaults to None.

        Returns:
            Any: The result of the torch function call.
        """

        def super_fn(*args: Any, **kwargs: Any):
            """Disable torch_function and returns the result of calling the `func` function with the given arguments and keyword arguments.

            Parameters:
                args (Any): Variable length argument list.
                kwargs (Any): Arbitrary keyword arguments.

            Returns:
                Any: The result of calling the `func` function with the given arguments and keyword arguments.
            """
            # Disable torch_function by hand because we don't want the wrapping behavior of
            # the super() impl
            # with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)

        if func in self.IMPLEMENTATIONS:
            return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})

        # This is just a no-op for all the non-factory functions:
        return super_fn(*args, **kwargs or {})


# Convenient wrapper to register functions
def implements_factory(func: Any):
    """
    Register a decorator function that implements a factory function.

    Args:
        func (Any): The factory function to register an implementation for.

    Returns:
        Callable: The decorated implementation function.
    """

    def _inner_fn(impl: Any):
        """
        Implement a decorator used to register an implementation for a specific function.

        Args:
            impl (Any): The implementation to be registered.

        Returns:
            Any: The registered implementation.
        """
        NPUDevice.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn


def parse_to_arguments(*args: Any, **kwargs: Any):
    """
    Parse the arguments and keyword arguments to handle device selection.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple: A tuple containing the following:
            - npu_device (bool): Indicates whether the device is an NPU device.
            - new_args (list): List of modified arguments.
            - kwargs (dict): Dictionary of modified keyword arguments.
    """
    device = kwargs.get("device", None)
    npu_device = False
    if device == "npu":
        npu_device = True
        kwargs["device"] = "cpu"

    new_args = []
    for arg in args:
        if arg == "npu":
            npu_device = True
            new_args.append("cpu")
        else:
            new_args.append(arg)

    return npu_device, new_args, kwargs


@implements_factory(torch.device)
def device(super_fn: Any, device, *args: Any, **kwargs: Any):
    """
    Return the device based on the input device name.

    Args:
        super_fn (Any): The super function to call.
        device (str): The name of the device.
        args (Any): Additional positional arguments to pass to the super function.
        kwargs (Any): Additional keyword arguments to pass to the super function.

    Returns:
        torch.device: The device object.

    """
    if device == "npu":
        # Patch the device to return the NPU device
        return torch.device("cpu")
    return super_fn(device, *args, **kwargs)


@implements_factory(torch.Tensor.to)
def to(super_fn: Any, self: Any, *args: Any, **kwargs: Any):
    """
    Convert the tensor to the specified device.

    Args:
        super_fn: The super function to call.
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        The converted tensor.

    Note:
        This implementation only supports a subset of the `.to()` functionality.
        Once the remote tensor feature is available, it can be converted to a remote tensor.
    """
    npu_device, args, kwargs = parse_to_arguments(*args, **kwargs)
    if npu_device:
        return super_fn(RemoteTensor.from_torch(self), *args, **kwargs)
    return super_fn(self, *args, **kwargs)


@implements_factory(torch._C._nn._parse_to)
def _parse_to(super_fn: Any, *args: Any, **kwarg: Any):
    """
    Parse the arguments and return the device, dtype, non_blocking, and convert_to_format.

    Args:
        super_fn (Any): The super function to call.
        args (Any): Positional arguments.
        kwarg (Any): Keyword arguments.

    Returns:
        Tuple: A tuple containing the device, dtype, non_blocking, and convert_to_format.
    """
    npu_device, args, kwargs = parse_to_arguments(*args, **kwarg)

    device, dtype, non_blocking, convert_to_format = super_fn(*args, **kwargs)

    if npu_device:
        device = "npu"

    return device, dtype, non_blocking, convert_to_format


def new_to(self, *args: Any, **kwargs: Any):
    """
    Move the input tensor(s) to the specified device.

    Args:
        args: Variable length argument list of devices to move the tensor(s) to.
        kwargs: Keyword arguments for the `to` method.

    Returns:
        Tensor or Module: The tensor or module with the tensor(s) moved to the specified device(s).
    """
    npu_device, args, kwargs = parse_to_arguments(*args, **kwargs)

    if npu_device:
        self = convert_to_npu_module(self).to("npu")

    return self._to(*args, **kwargs)


@lru_cache()
def enable_npu_device():
    """
    Enable the NPU device for acceleration.

    This function globally enables the NPU device mode by creating an instance of `NPUDevice` and
    modifying the `torch.nn.Module.to` method to use a custom implementation called `new_to`.

    Usage:
        enable_npu_device()

    """
    holder = NPUDevice()
    holder.__enter__()
    torch.nn.Module._to = torch.nn.Module.to
    torch.nn.Module.to = new_to

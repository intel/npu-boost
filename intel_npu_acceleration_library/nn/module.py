#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend import NNFactory, Tensor
from typing import MutableMapping, Sequence, Any, List
from torch.profiler import record_function
import numpy as np
import torch


def pt_to_np_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """Convert a PyTorch dtype to a NumPy dtype.

    Args:
        torch_dtype (torch.dtype): The PyTorch dtype to convert.

    Raises:
        ValueError: If the PyTorch dtype is not supported.

    Returns:
        np.dtype: The NumPy dtype.
    """
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.int16:
        return np.int16
    elif torch_dtype == torch.int32:
        return np.int32
    elif torch_dtype == torch.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype {torch_dtype}")


def compute_input_signature(
    args: Sequence[Any], kwargs: MutableMapping[str, Any]
) -> str:
    """Compute the input signature of a function call.

    Args:
        args (Sequence[Any]): The positional arguments.
        kwargs (MutableMapping[str, Any]): The keyword arguments.

    Returns:
        str: The input signature.
    """
    signature = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            signature.append("_".join(str(dim) for dim in arg.shape))
            signature.append(str(arg.dtype))
        else:
            signature.append(str(arg))
    for k, arg in kwargs.items():
        if isinstance(arg, torch.Tensor):
            signature.append(str(k))
            signature.append("_".join(str(dim) for dim in arg.shape))
            signature.append(str(arg.dtype))
        else:
            signature.append(str(arg))
    return "_".join(signature)


def patch_modules(module: torch.nn.Module, model: NNFactory):
    """Patch the modules of a PyTorch module with constants.

    Args:
        module (torch.nn.Module): The PyTorch module.
        model (NNFactory): The NNFactory instance.
    """
    modules = list(module.named_children())
    for _, module in modules:
        if isinstance(module, Module):
            module.npu_top_level_module = False
        patch_modules(module, model)


class Module(torch.nn.Module):
    """A PyTorch module that runs on the NPU."""

    def __init__(self, profile: bool = False) -> None:
        """Initialize the module.

        Args:
            profile (bool): Enable model profiling. Defaults to False.
        """
        super().__init__()
        self._nn_factory_cache: MutableMapping[str, NNFactory] = {}
        self._npu_inference = False
        self.npu_top_level_module = True
        self.profile = profile

    def extract_tensors_from_arguments(
        self, args: Sequence[Any]
    ) -> Sequence[torch.Tensor]:
        """Extract the tensors from the arguments.

        Args:
            args (Sequence[Any]): The positional arguments.

        Returns:
            Sequence[torch.Tensor]: The tensors.
        """
        tensors, non_tensors = [], []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors.append(arg)
            elif isinstance(arg, (list, tuple)):
                tensor_list, non_tensor_list = self.extract_tensors_from_arguments(arg)
                tensors.extend(tensor_list)
                non_tensors.extend(non_tensor_list)
            elif isinstance(arg, dict):
                tensor_list, non_tensor_list = self.extract_tensors_from_arguments(
                    list(arg.values())
                )
                tensors.extend(tensor_list)
                non_tensors.extend(non_tensor_list)
        return tensors, non_tensors

    def factory_forward(self, *args: Any, **kwargs: Any):
        """Run the model using the factory.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        signature = compute_input_signature(args, kwargs)
        model = self._nn_factory_cache[signature]

        tensor_args, non_tensor_args = self.extract_tensors_from_arguments(args)
        tensor_args.extend(
            self.extract_tensors_from_arguments(list(kwargs.values()))[0]
        )

        return model(*tensor_args, *non_tensor_args, **kwargs)

    def create_model(
        self, args: Sequence[Any], kwargs: MutableMapping[str, Any]
    ) -> NNFactory:
        """Create a model from the module.

        Args:
            args (Sequence[Any]): positional arguments
            kwargs (MutableMapping[str, Any]): keyword arguments

        Returns:
            NNFactory: The model.
        """
        model = NNFactory(profile=self.profile)

        def create_args_from_list(args: Sequence[Any]) -> Sequence[Any]:
            """Create arguments from a list.

            Args:
                args (Sequence[Any]): The arguments.

            Returns:
                Sequence[Any]: The npu converted arguments.
            """
            npu_args: List[Any] = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    npu_args.append(
                        model.parameter(arg.shape, pt_to_np_dtype(arg.dtype))
                    )
                elif isinstance(arg, (list, tuple)):
                    npu_args.append(create_args_from_list(arg))
                elif isinstance(arg, dict):
                    npu_args.append(create_kwargs_from_list(arg))
                else:
                    npu_args.append(arg)
            return npu_args

        def create_kwargs_from_list(
            kwargs: MutableMapping[str, Any]
        ) -> MutableMapping[str, Any]:
            """Create keyword arguments from a list.

            Args:
                kwargs (MutableMapping[str, Any]): The keyword arguments.

            Returns:
                MutableMapping[str, Any]: The npu converted keyword arguments.
            """
            npu_kwargs: MutableMapping[str, Any] = {}
            for k, arg in kwargs.items():
                if isinstance(arg, torch.Tensor):
                    npu_kwargs[k] = model.parameter(
                        arg.shape, pt_to_np_dtype(arg.dtype)
                    )
                elif isinstance(arg, (list, tuple)):
                    npu_kwargs[k] = create_args_from_list(arg)
                elif isinstance(arg, dict):
                    npu_kwargs[k] = create_kwargs_from_list(arg)
                else:
                    npu_kwargs[k] = arg
            return npu_kwargs

        npu_args = create_args_from_list(args)
        npu_kwargs = create_kwargs_from_list(kwargs)

        patch_modules(self, model)

        _ = self.forward(*npu_args, **npu_kwargs)
        model.compile()
        return model

    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            Any: The output of the module.
        """
        if self._npu_inference and self.npu_top_level_module:

            signature = compute_input_signature(args, kwargs)
            if signature not in self._nn_factory_cache:
                self._nn_factory_cache[signature] = self.create_model(args, kwargs)

            # Run the model by replacing the forward method with the factory_forward
            old_forward = self.forward
            self.forward = self.factory_forward  # type: ignore
            with record_function(f"npu_{self.__class__.__name__}"):
                out = super()._call_impl(*args, **kwargs)

            # Restore the original forward method
            self.forward = old_forward  # type: ignore

            return out
        else:
            return super()._call_impl(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Move the module to a device or to a different dtype.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        device = kwargs.get("device", None)
        args = list(args)
        if device is None:
            for idx, arg in enumerate(args):
                if isinstance(arg, str) and arg.lower() in ["npu"]:
                    device = "npu"
                    args[idx] = "cpu"
        else:
            kwargs["device"] = "cpu"

        if device.lower() == "npu":
            self._npu_inference = True

        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the forward pass of the module.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Raises:
            NotImplementedError: If the forward method is not implemented.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError
        return torch.empty(0)


class NPUModuleWrapper(Module):
    """A PyTorch module that runs on the NPU."""

    def __init__(self, module: torch.nn.Module) -> None:
        """Initialize the module.

        Args:
            module (torch.nn.Module): The PyTorch module.
        """
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the forward pass of the module.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        with record_function(f"npu_{self.module.__class__.__name__}"):
            return self.module(*args, **kwargs)


def convert_to_npu_module(module: torch.nn.Module) -> Module:
    """Convert a PyTorch module to an NPU Module.

    Args:
        module (torch.nn.Module): The PyTorch module.

    Returns:
        Module: The NPU enabled Module.
    """
    return NPUModuleWrapper(module).eval()


class NPUContextManager(NNFactory):
    """NPU context manager."""

    def __enter__(self):
        """Enter the context.

        Returns:
            NPUContextManager: self
        """
        return self

    def Constant(self, tensor: torch.Tensor) -> Tensor:
        """Create a tensor.

        Args:
            tensor (torch.Tensor): tensor

        Returns:
            torch.Tensor: tensor
        """
        return self.constant(tensor)  # type: ignore

    def Tensor(
        self, shape: Sequence[int], dtype: torch.dtype = torch.float16
    ) -> Tensor:
        """Create a tensor.

        Args:
            shape (Sequence[int]): tensor shape
            dtype (torch.dtype): tensor dtype, default to torch.float16

        Returns:
            Tensor: tensor
        """
        return self.parameter(shape, dtype=dtype)  # type: ignore

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context.

        Args:
            exc_type: exception type
            exc_value: exception value
            traceback: traceback

        Raises:
            RuntimeError: If an exception is raised.
        """
        # If there is no exception, call the compile
        if exc_type is None:
            self.compile()
        else:
            # raise the exception
            print(exc_type, exc_value, traceback)
            raise RuntimeError(exc_value)  # .with_traceback(traceback)

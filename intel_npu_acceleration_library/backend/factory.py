#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.base import BaseNPUBackendWithPrefetch
from intel_npu_acceleration_library.backend.ops import get_supported_ops
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from intel_npu_acceleration_library.backend.tensor import Tensor
from intel_npu_acceleration_library.dtypes import get_backend_dtype
from typing import Optional, Tuple, Any, Union, Sequence, TypeVar, Callable, cast, List
from functools import partial
import numpy.typing as npt
import numpy as np
import ctypes
import torch


F = TypeVar("F", bound=Callable[..., Any])


class NNFactory(BaseNPUBackendWithPrefetch):
    """Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the Linear class.

        Args:
            profile (Optional[bool], optional): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
        """
        super().__init__(profile)
        self.device = device
        self._mm = backend_lib.createNNFactory(
            ctypes.c_char_p(self.device.encode()),
            profile,
        )
        self.elapsed = None
        self.output_nodes: Sequence[ctypes._Pointer] = []

        for op in get_supported_ops():
            if not hasattr(self, op.name.replace("_act", "")):
                setattr(
                    self,
                    op.name.replace("_act", ""),
                    partial(self._call_backend_op, op.name),
                )

    def return_tensor(fn: F) -> F:  # type: ignore
        """Wrap the output of a function in a Tensor object.

        Args:
            fn (function): Function

        Returns:
            function: A function that wraps the output in a Tensor object
        """

        def wrapper(self, *args: Any, **kwargs: Any) -> Tensor:
            """Wrap the output of a function in a Tensor object.

            Args:
                args (Any): Variable length argument list
                kwargs (Any): Arbitrary keyword arguments

            Returns:
                Tensor: Tensor object
            """
            # Convert Tensor objects to their underlying node
            args = tuple(arg.node if isinstance(arg, Tensor) else arg for arg in args)
            kwargs = {
                k: v.node if isinstance(v, Tensor) else v for k, v in kwargs.items()
            }

            input_nodes = [arg for arg in args if isinstance(arg, ctypes._Pointer)] + [
                v for v in kwargs.values() if isinstance(v, ctypes._Pointer)
            ]
            # Call the function
            node = fn(self, *args, **kwargs)

            # remove input nodes from output_nodes
            self.output_nodes = [
                node for node in self.output_nodes if node not in input_nodes
            ]
            # add output node to output_nodes
            if fn.__name__ != "constant":
                self.output_nodes.append(node)

            # Wrap the node in a Tensor object
            return Tensor(factory=self, node=node)

        return cast(F, wrapper)

    @return_tensor
    def _call_backend_op(self, op_name: str, *parameters: Any) -> Any:
        """Dynamically call a backend operation.

        Args:
            op_name (str): operation name
            parameters (Any): variable list of operation parameters

        Returns:
            Any: Operation
        """
        fn = getattr(backend_lib, op_name)
        return fn(self._mm, *parameters)

    def get_backend_dtype(self, dtype) -> ctypes.c_char_p:
        """Get the string representation of the dtype.

        Args:
            dtype: numpy dtype

        Returns:
            ctypes.c_char_p: string representation of the dtype
        """
        return get_backend_dtype(dtype)

    @return_tensor
    def parameter(
        self, shape: Sequence[int], dtype: npt.DTypeLike = np.float16
    ) -> ctypes._Pointer:
        """Generate a model input parameter.

        Args:
            shape (Sequence[int]): Parameter shape
            dtype (np.dtype, optional): parameter type np.int8, np.uint8 and np.float16 supported. Defaults to np.float16. Unit8 represents packed i4 dtypes

        Returns:
            ctypes._Pointer: an instance to a parameter object

        """
        shape_ptr = np.array(shape, dtype=np.uint32)
        return backend_lib.parameter(
            self._mm, shape_ptr.size, shape_ptr, self.get_backend_dtype(dtype)
        )

    @return_tensor
    def to(self, tensor: ctypes._Pointer, dtype: npt.DTypeLike) -> ctypes._Pointer:
        """Convert a tensor to a different dtype.

        Args:
            tensor (ctypes._Pointer): input tensor
            dtype (npt.DTypeLike): target dtype

        Returns:
            ctypes._Pointer: output tensor
        """
        dtype_ptr = self.get_backend_dtype(dtype)
        return backend_lib.to(self._mm, tensor, dtype_ptr)

    @return_tensor
    def constant(
        self,
        data: Union[np.array, Sequence[int], Sequence[float], int, float, torch.Tensor],
    ) -> ctypes._Pointer:
        """Generate a model input constant.

        Args:
            data (Union[np.array, Sequence[int], Sequence[float], int, float, torch.Tensor]): constant data

        Returns:
            ctypes._Pointer: an instance to a constant object

        """
        if isinstance(data, (list, tuple)):
            if all(isinstance(i, int) for i in data):
                data = np.array(data, dtype=np.int64)
            else:
                data = np.array(data, dtype=np.float32)
        elif isinstance(data, int):
            data = np.array([data], dtype=np.int64)
        elif isinstance(data, float):
            data = np.array([data], dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        elif data is None:
            return ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_char))

        dst = data.ctypes.data_as(ctypes.c_void_p)
        shape_ptr = np.array(data.shape, dtype=np.uint32)
        return backend_lib.constant(
            self._mm, shape_ptr.size, shape_ptr, self.get_backend_dtype(data.dtype), dst
        )

    @return_tensor
    def matmul(
        self,
        input_node: ctypes._Pointer,
        weights_node: ctypes._Pointer,
        trA: bool = False,
        trB: bool = True,
    ) -> ctypes._Pointer:
        """Generate a matrix multiplication layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            weights_node (ctypes._Pointer): weights node
            trA (bool): transpose input node
            trB (bool): transpose weights node

        Returns:
            ctypes._Pointer: output node
        """
        return backend_lib.matmul(self._mm, input_node, weights_node, trA, trB)

    @return_tensor
    def convolution(
        self,
        input_node: ctypes._Pointer,
        weights_node: ctypes._Pointer,
        bias: Optional[ctypes._Pointer] = None,
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        act_dtype: npt.DTypeLike = np.float16,
        n_spatial_dims: int = 2,
    ) -> ctypes._Pointer:
        """Generate a convolution layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            weights_node (ctypes._Pointer): weights node
            bias (Optional[ctypes._Pointer}): bias node
            strides (Sequence[int]): strides
            padding (Sequence[int]): padding
            dilation (Sequence[int]): dilation
            groups (int): groups
            act_dtype (npt.DTypeLike, optional): activation dtype. Defaults to np.float16.
            n_spatial_dims (int): number of spatial dimensions

        Returns:
            ctypes._Pointer: output node
        """
        if isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        if isinstance(dilation, int):
            dilation = [dilation] * n_spatial_dims

        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        dilation_ptr = np.array(dilation, dtype=np.uint32)

        if bias is not None:
            bias_node = bias
        else:
            bias_node = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_char))

        return backend_lib.convolution(
            self._mm,
            input_node,
            weights_node,
            bias_node,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            dilation_ptr.size,
            dilation_ptr,
            groups,
            self.get_backend_dtype(act_dtype),
        )

    @return_tensor
    def linear(
        self,
        input_node: ctypes._Pointer,
        output_channels: int,
        input_channels: int,
        bias: Optional[bool] = False,
        act_dtype: npt.DTypeLike = np.float16,
        wt_dtype: npt.DTypeLike = np.float16,
    ) -> ctypes._Pointer:
        """Generate a linear layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            output_channels (int): number of output channels
            input_channels (int): number of input channels
            bias (bool, optional): enable/disable bias. Defaults to False.
            act_dtype (npt.DTypeLike, optional): activation dtype. Defaults to np.float16.
            wt_dtype (npt.DTypeLike, optional): weight dtype. Defaults to np.float16.

        Returns:
            ctypes._Pointer: output node
        """
        return backend_lib.linear(
            self._mm,
            input_node,
            output_channels,
            input_channels,
            bias,
            self.get_backend_dtype(act_dtype),
            self.get_backend_dtype(wt_dtype),
        )

    @return_tensor
    def reshape(
        self, input_node: ctypes._Pointer, shape: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate a reshape layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            shape (Sequence[int]): shape

        Returns:
            ctypes._Pointer: output node
        """
        shape_node = self.constant(shape).node  # type: ignore
        return backend_lib.reshape(self._mm, input_node, shape_node)

    @return_tensor
    def transpose(
        self, input_node: ctypes._Pointer, input_order: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate a transpose layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            input_order (Sequence[int]): input order

        Returns:
            ctypes._Pointer: output node
        """
        input_order_node = self.constant(input_order).node  # type: ignore
        return backend_lib.transpose(self._mm, input_node, input_order_node)

    @return_tensor
    def unsqueeze(
        self, input_node: ctypes._Pointer, axis: Sequence[int]
    ) -> ctypes._Pointer:
        """Generate an unsqueeze layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            axis (Sequence[int]): axis

        Returns:
            ctypes._Pointer: output node
        """
        axis_node = self.constant(axis).node  # type: ignore
        return backend_lib.unsqueeze(self._mm, input_node, axis_node)

    @return_tensor
    def slice(
        self,
        input_node: ctypes._Pointer,
        begin: Sequence[int],
        end: Sequence[int],
        stride: Optional[Sequence[int]] = None,
    ) -> ctypes._Pointer:
        """Generate an unsqueeze layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            begin (Sequence[int]): begin
            end (Sequence[int]): end
            stride (Optional[Sequence[int]]): stride

        Raises:
            ValueError: begin and end must have the same length

        Returns:
            ctypes._Pointer: output node
        """
        if len(begin) != len(end):
            raise ValueError("begin and end must have the same length")

        if stride is None:
            stride = [1] * len(begin)

        begin_mask_ptr = np.zeros([len(begin)], dtype=np.uint32)
        end_mask_ptr = np.zeros([len(begin)], dtype=np.uint32)

        begin = self.constant(begin).node  # type: ignore
        end = self.constant(end).node  # type: ignore
        stride = self.constant(stride).node  # type: ignore

        return backend_lib.slice(
            self._mm,
            input_node,
            begin,
            end,
            stride,
            begin_mask_ptr.size,
            begin_mask_ptr,
            end_mask_ptr.size,
            end_mask_ptr,
        )

    @return_tensor
    def concat(
        self, input_node_1: ctypes._Pointer, input_node_2: ctypes._Pointer, axis: int
    ) -> ctypes._Pointer:
        """Generate a concatenation layer.

        Args:
            input_node_1 (ctypes._Pointer): first layer input node
            input_node_2 (ctypes._Pointer): second layer input node
            axis (int): axis

        Returns:
            ctypes._Pointer: output node
        """
        if axis < 0:
            shape_size = backend_lib.op_shape_size(input_node_1)
            axis = (axis + shape_size) % shape_size
        axis = np.int64(axis)
        return backend_lib.concat(self._mm, input_node_1, input_node_2, axis)

    @return_tensor
    def reduce_max(
        self,
        input_node: ctypes._Pointer,
        reduction_axes: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
    ) -> ctypes._Pointer:
        """Generate a reduce max layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            reduction_axes (Optional[Union[int, Sequence[int]]]): the axis positions to be reduced
            keep_dims (Optional[bool]): if set to 1 it holds axes that are used for reduction. Defaults to False

        Returns:
            ctypes._Pointer: output node
        """
        if reduction_axes is None:
            shape_size = backend_lib.op_shape_size(input_node)
            reduction_axes = list(range(shape_size - 1, -1, -1))
        axis_node = self.constant(reduction_axes).node  # type: ignore
        return backend_lib.reduce_max(self._mm, input_node, axis_node, keep_dims)

    @return_tensor
    def reduce_mean(
        self,
        input_node: ctypes._Pointer,
        reduction_axes: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
    ) -> ctypes._Pointer:
        """Generate a reduce mean layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            reduction_axes (Optional[Union[int, Sequence[int]]]): the axis positions to be reduced
            keep_dims (Optional[bool] ): if set to 1 it holds axes that are used for reduction. Defaults to False

        Returns:
            ctypes._Pointer: output node
        """
        if reduction_axes is None:
            shape_size = backend_lib.op_shape_size(input_node)
            reduction_axes = list(range(shape_size - 1, -1, -1))
        axis_node = self.constant(reduction_axes).node  # type: ignore
        return backend_lib.reduce_mean(self._mm, input_node, axis_node, keep_dims)

    @return_tensor
    def reduce_min(
        self,
        input_node: ctypes._Pointer,
        reduction_axes: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
    ) -> ctypes._Pointer:
        """Generate a reduce min layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            reduction_axes (Optional[Union[int, Sequence[int]]]): the axis positions to be reduced
            keep_dims (Optional[bool] ): if set to 1 it holds axes that are used for reduction. Defaults to False

        Returns:
            ctypes._Pointer: output node
        """
        if reduction_axes is None:
            shape_size = backend_lib.op_shape_size(input_node)
            reduction_axes = list(range(shape_size - 1, -1, -1))
        axis_node = self.constant(reduction_axes).node  # type: ignore
        return backend_lib.reduce_min(self._mm, input_node, axis_node, keep_dims)

    @return_tensor
    def reduce_prod(
        self,
        input_node: ctypes._Pointer,
        reduction_axes: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
    ) -> ctypes._Pointer:
        """Generate a reduce product layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            reduction_axes (Optional[Union[int, Sequence[int]]]): the axis positions to be reduced
            keep_dims (Optional[bool] ): if set to 1 it holds axes that are used for reduction. Defaults to False

        Returns:
            ctypes._Pointer: output node
        """
        if reduction_axes is None:
            shape_size = backend_lib.op_shape_size(input_node)
            reduction_axes = list(range(shape_size - 1, -1, -1))
        axis_node = self.constant(reduction_axes).node  # type: ignore
        return backend_lib.reduce_prod(self._mm, input_node, axis_node, keep_dims)

    @return_tensor
    def reduce_sum(
        self,
        input_node: ctypes._Pointer,
        reduction_axes: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
    ) -> ctypes._Pointer:
        """Generate a reduce sum layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            reduction_axes (Optional[Union[int, Sequence[int]]]): the axis positions to be reduced
            keep_dims (Optional[bool] ): if set to 1 it holds axes that are used for reduction. Defaults to False

        Returns:
            ctypes._Pointer: output node
        """
        if reduction_axes is None:
            shape_size = backend_lib.op_shape_size(input_node)
            reduction_axes = list(range(shape_size - 1, -1, -1))
        axis_node = self.constant(reduction_axes).node  # type: ignore
        return backend_lib.reduce_sum(self._mm, input_node, axis_node, keep_dims)

    @return_tensor
    def normL2(
        self, input_node: ctypes._Pointer, axis: int, eps: Optional[float] = 1e-12
    ) -> ctypes._Pointer:
        """Generate an L2 normalization layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            axis (int): axis
            eps (float, optional): epsilon added to L2 norm. Defaults to 1e-12

        Returns:
            ctypes._Pointer: output node
        """
        if axis < 0:
            shape_size = backend_lib.op_shape_size(input_node)
            axis = (axis + shape_size) % shape_size
        axis_node = self.constant(axis).node  # type: ignore
        return backend_lib.normL2(self._mm, input_node, axis_node, eps)

    @return_tensor
    def power(
        self,
        input_node: ctypes._Pointer,
        exponent: Union[ctypes._Pointer, torch.Tensor],
    ) -> ctypes._Pointer:
        """Generate a power layer.

        Args:
            input_node (ctypes._Pointer): layer input node
            exponent (Union[ctypes._Pointer, torch.Tensor]): the exponent value

        Raises:
            ValueError: Input tensor shapes are not equal

        Returns:
            ctypes._Pointer: output node
        """
        input_shape_size = backend_lib.op_shape_size(input_node)
        input_shape = [
            backend_lib.op_shape(input_node, i) for i in range(input_shape_size)
        ]
        if isinstance(exponent, ctypes._Pointer):
            exponent_shape_size = backend_lib.op_shape_size(input_node)
            exponent_shape = [
                backend_lib.op_shape(exponent, i) for i in range(exponent_shape_size)
            ]
        else:
            exponent_shape = list(exponent.shape)
            exponent = self.constant(exponent).node  # type: ignore
        if exponent_shape != input_shape:
            raise ValueError("Input tensor shapes are not equal")

        return backend_lib.power(self._mm, input_node, exponent)

    @return_tensor
    def avg_pooling(
        self,
        input: ctypes._Pointer,
        kernel_size: Union[int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        n_spatial_dims: int = 2,
    ) -> ctypes._Pointer:
        """Generate an average pooling layer.

        Args:
            input (ctypes._Pointer): layer input node
            kernel_size (Sequence[int]): kernel size
            strides (Sequence[int]): strides
            padding (int): padding
            ceil_mode (bool): ceil mode
            count_include_pad (bool): count include pad
            divisor_override (int): divisor override
            n_spatial_dims (int): number of spatial dimensions

        Raises:
            NotImplementedError: divisor_override is not supported

        Returns:
            ctypes._Pointer: output node
        """
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * n_spatial_dims

        if strides is None:
            strides = kernel_size
        elif isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        kernel_size_ptr = np.array(kernel_size, dtype=np.uint32)

        rounding_type = 1 if ceil_mode else 0
        auto_pad = 0  # Hardcoded to explicit padding

        if divisor_override:
            raise NotImplementedError("divisor_override is not supported")

        return backend_lib.avg_pooling(
            self._mm,
            input,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            kernel_size_ptr.size,
            kernel_size_ptr,
            not count_include_pad,  # exclude_pad
            rounding_type,  # rounding_type
            auto_pad,  # auto_pad
        )

    @return_tensor
    def max_pooling(
        self,
        input: ctypes._Pointer,
        kernel_size: Union[int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        n_spatial_dims: int = 2,
    ) -> ctypes._Pointer:
        """Generate an average pooling layer.

        Args:
            input (ctypes._Pointer): layer input node
            kernel_size (Sequence[int]): kernel size
            strides (Sequence[int]): strides
            padding (int): padding
            ceil_mode (bool): ceil mode
            n_spatial_dims (int): number of spatial dimensions

        Returns:
            ctypes._Pointer: output node
        """
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * n_spatial_dims

        if strides is None:
            strides = kernel_size
        elif isinstance(strides, int):
            strides = [strides] * n_spatial_dims

        if isinstance(padding, int):
            padding_begins = [padding] * n_spatial_dims
            padding_ends = [padding] * n_spatial_dims
        else:
            padding_begins = list(padding)
            padding_ends = list(padding)

        strides_ptr = np.array(strides, dtype=np.uint32)
        padding_begins_ptr = np.array(padding_begins, dtype=np.uint32)
        padding_ends_ptr = np.array(padding_ends, dtype=np.uint32)
        kernel_size_ptr = np.array(kernel_size, dtype=np.uint32)

        rounding_type = 1 if ceil_mode else 0
        auto_pad = 0  # Hardcoded to explicit padding

        return backend_lib.max_pooling(
            self._mm,
            input,
            strides_ptr.size,
            strides_ptr,
            padding_begins_ptr.size,
            padding_begins_ptr,
            padding_ends_ptr.size,
            padding_ends_ptr,
            kernel_size_ptr.size,
            kernel_size_ptr,
            rounding_type,  # rounding_type
            auto_pad,  # auto_pad
        )

    def get_tensor_shape(self, node):
        """Get tensor shape.

        Args:
            node: network node

        Returns:
            tuple[int]: tensor shape
        """
        size = backend_lib.op_shape_size(node)
        return tuple([backend_lib.op_shape(node, idx) for idx in range(size)])

    def get_tensor_dtype(self, node):
        """Get tensor dtype.

        Args:
            node: network node

        Raises:
            RuntimeError: Unsupported dtype

        Returns:
            str: tensor dtype
        """
        dtype_int = backend_lib.op_dtype(node)

        if dtype_int == 2:
            return np.bool
        # elif dtype_int == 3:
        #     return bfloat16
        elif dtype_int == 4:
            return np.float16
        elif dtype_int == 5:
            return np.float32
        elif dtype_int == 6:
            return np.float64
        # elif dtype_int == 7:
        #     return int4
        elif dtype_int == 8:
            return np.int8
        elif dtype_int == 9:
            return np.int16
        elif dtype_int == 10:
            return np.int32
        elif dtype_int == 11:
            return np.int64
        else:
            raise RuntimeError("Unsupported dtype")

    def compile(self):
        """Finalize and compile a model."""
        self.out = []
        for node in self.output_nodes:
            backend_lib.result(self._mm, node)

        # Compile the model
        backend_lib.compile(self._mm)

        for idx, node in enumerate(self.output_nodes):
            output_shape = self.get_tensor_shape(node)
            output_dtype = self.get_tensor_dtype(node)

            tensor = np.empty(output_shape, dtype=output_dtype)
            ptr = tensor.ctypes.data_as(ctypes.c_void_p)
            backend_lib.set_output(self._mm, ptr, idx)
            self.out.append(tensor)

    def set_input_tensor(self, tensor: np.ndarray, idx: int):
        """Set input tensor.

        Args:
            tensor (np.ndarray): Input tensor
            idx (int): tensor index
        """
        backend_lib.set_activation(
            self._mm, tensor.ctypes.data_as(ctypes.c_void_p), idx
        )

    def get_tensor_recursively(self, args: Sequence[Any]) -> List[np.ndarray]:
        """Get tensor recursively for a list of arguments.

        Args:
            args (Sequence[Any]): Sequence of tensors, tuple of tensors and additional arguments

        Returns:
            List[np.ndarray]: Sequence of tensors
        """
        tensors = []
        for t in args:
            if isinstance(t, (list, tuple)):
                tensors.extend(self.get_tensor_recursively(t))
            elif isinstance(t, np.ndarray):
                tensors.append(t)

        return tensors

    def run(
        self,
        X: np.ndarray,
        *weights: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the layer: X * W^T.

        Args:
            X (np.ndarray): lhs operator
            weights (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): rhs operators
            kwargs (Any): additional arguments

        Returns:
            np.ndarray: result
        """
        op_id = kwargs.get("op_id", None)
        if op_id is None:
            ww = self.get_tensor_recursively(weights)
            for idx, weight in enumerate(ww):
                self.set_input_tensor(weight, idx + 1)
            prefetch = False
        else:
            prefetch = self.setWeights(kwargs.get("op_id", None), *weights)

        self.set_input_tensor(X, 0)
        self.elapsed = backend_lib.run(self._mm)

        if prefetch:
            self.prefetchWeights()

        if len(self.out) == 1:
            return self.out[0]
        return self.out

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Run the model using the factory.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            np.ndarray: The output tensor.
        """
        args = tuple(
            [
                arg.detach().numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
        )
        kwargs = {
            k: arg.detach().numpy() if isinstance(arg, torch.Tensor) else arg
            for k, arg in kwargs.items()
        }

        out = self.run(*args, **kwargs)
        if isinstance(out, list):
            return [torch.tensor(o, device=torch.device("npu")) for o in out]
        return torch.tensor(out, device=torch.device("npu"))

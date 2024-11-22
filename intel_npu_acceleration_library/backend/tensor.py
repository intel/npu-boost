#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend import lib as backend_lib
from typing import Sequence, Any, Optional, MutableMapping, Union
from intel_npu_acceleration_library.dtypes import (
    float16,
    bfloat16,
    float32,
    float64,
    int4,
    int8,
    int16,
    int32,
    int64,
    NPUDtype,
    get_backend_dtype,
)
from dataclasses import dataclass
import functools
from math import prod
import numpy as np
import ctypes
import torch


class RemoteTensor(torch.Tensor):
    """
    Represent a remote tensor object.

    Attrs:
        _remote_tensor (ctypes._Pointer): The pointer to the underlying remote tensor.

    Methods:
        from_torch(x: torch.Tensor): Create a remote tensor from a torch tensor.
    """

    _remote_tensor = None

    @staticmethod
    def __new__(cls, x: Any, remote_tensor: ctypes._Pointer, *args: Any, **kwargs: Any):
        """
        Create a new remote tensor object.

        Args:
            x (Any): tensor input
            remote_tensor (ctypes._Pointer): remote tensor pointer
            args (Any): additional arguments
            kwargs (Any): additional keyword arguments

        Returns:
            RemoteTensor: a RemoteTensor object
        """
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x: Any, remote_tensor: ctypes._Pointer):
        """
        Initialize the remote tensor object.

        Args:
            x (Any): tensor input
            remote_tensor (ctypes._Pointer): remote tensor pointer
        """
        self._remote_tensor = remote_tensor

    # def __del__(self):
    #     if self._remote_tensor and backend_lib:
    #         backend_lib.del_remote_tensor(self._remote_tensor)

    @staticmethod
    def from_torch(x: torch.Tensor) -> "RemoteTensor":
        """
        Create a remote tensor from a torch tensor.

        Args:
            x (torch.Tensor): The torch tensor.

        Returns:
            RemoteTensor: The remote tensor.
        """
        shape_arr = np.array(x.shape, dtype=np.uint32)
        dtype_str = get_backend_dtype(x.dtype)
        p = ctypes.cast(x.data_ptr(), ctypes.c_void_p)

        rt = backend_lib.to_npu(shape_arr.size, shape_arr, dtype_str, p)

        pointer = ctypes.cast(
            backend_lib.remote_tensor_data(rt),
            ctypes.POINTER(ctypes.c_uint8),
        )

        arr = (pointer._type_ * prod(x.shape) * x.element_size()).from_address(
            ctypes.addressof(pointer.contents)
        )

        pt_tensor = torch.frombuffer(arr, dtype=x.dtype).view(*x.shape)

        return RemoteTensor(pt_tensor, rt)


@dataclass
class Tensor:
    """
    Represents a tensor object.

    Attrs:
        factory (NNFactory): The factory object used to create the tensor.
        node (ctypes._Pointer): The pointer to the underlying tensor node.
        shape (Sequence[int]): The shape of the tensor.
        dtype (NPUDtype): The data type of the tensor.
        T (Tensor): The transpose of the tensor.

    Methods:
        __add__(self, other): Adds two tensors element-wise.
        __sub__(self, other): Subtracts two tensors element-wise.
        __mul__(self, other): Multiplies two tensors element-wise.
        __truediv__(self, other): Divides two tensors element-wise.
        __neg__(self): Negates the tensor.
        __repr__(self): Returns a string representation of the tensor.
        __str__(self): Returns a string representation of the tensor.
        __len__(self): Returns the total number of elements in the tensor.
        T: Returns the transpose of the tensor.
        squeeze(self): Removes dimensions of size 1 from the tensor.
        unsqueeze(self, axis): Adds a dimension of size 1 to the tensor.
        __matmul__(self, other): Performs matrix multiplication between two tensors.
        acos(self): Applies acos function to the tensor.
        asin(self): Applies asin function to the tensor.
        atan(self): Applies atan function to the tensor.
        acosh(self): Applies acosh function to the tensor.
        asinh(self): Applies asinh function to the tensor.
        atanh(self): Applies atanh function to the tensor.
        cosh(self): Applies cosh function to the tensor.
        sinh(self): Applies sinh function to the tensor.
        tanh(self): Applies tanh function to the tensor.
        cos(self): Applies cos function to the tensor.
        sin(self): Applies sin function to the tensor.
        tan(self): Applies tan function to the tensor.
        ceiling(self): Applies ceil function to the tensor.
        clamp(self, min, max): Applies clamp function to the tensor.
        elu(self, alpha): Applies elu function to the tensor.
        erf(self): Applies erf function to the tensor.
        exp(self): Applies exponental function to the tensor.
        floor(self): Applies floor function to the tensor.
        grn(self, bias): Applies grn function to the tensor.
        hsigmoid(self): Applies hsigmoid function to the tensor.
        hswish(self): Applies hswish function to the tensor.
        log(self): Applies log function to the tensor.
        mish(self): Applies mish function to the tensor.
        relu(self, bias): Applies relu function to the tensor.
        round(self): Applies round function to the tensor.
        sigmoid(self): Applies sigmoid function to the tensor.
        sign(self): Applies sign function to the tensor.
        softmax(self, dim): Applies softmax function to the tensor.
        softplus(self): Applies softplus function to the tensor.
        sqrt(self): Applies sqrt function to the tensor.
        max(self, dim, keep_dims): Returns the reduced max tensor.
        mean(self, dim, keep_dims, dtype): Returns the reduced mean tensor.
        min(self, dim, keep_dims): Returns the reduced min tensor.
        prod(self, dim, keep_dims, dtype): Returns the reduced product tensor.
        sum(self, dim, keep_dims, dtype): Returns the reduced sum tensor.
    """

    factory: "NNFactory"  # type: ignore # noqa: F821
    node: ctypes._Pointer

    @property
    def shape(self) -> Sequence[int]:
        """
        Returns the shape of the tensor.

        Returns:
            Sequence[int]: The shape of the tensor.
        """
        shape_size = backend_lib.op_shape_size(self.node)
        return [backend_lib.op_shape(self.node, i) for i in range(shape_size)]

    @property
    def dtype(self) -> NPUDtype:
        """
        Returns the data type of the tensor.

        Returns:
            type: The data type of the tensor.
        """
        dtype_int = backend_lib.op_dtype(self.node)

        if dtype_int == 2:
            return np.bool
        elif dtype_int == 3:
            return bfloat16
        elif dtype_int == 4:
            return float16
        elif dtype_int == 5:
            return float32
        elif dtype_int == 6:
            return float64
        elif dtype_int == 7:
            return int4
        elif dtype_int == 8:
            return int8
        elif dtype_int == 9:
            return int16
        elif dtype_int == 10:
            return int32
        elif dtype_int == 11:
            return int64
        else:
            raise RuntimeError("Unsupported dtype")

    def dim(self) -> int:
        """
        Return the number of dimensions of the tensor.

        Returns:
            int: The number of dimensions of the tensor.
        """
        return len(self.shape)

    def size(self, dim=None) -> Union[int, Sequence[int]]:
        """
        Return the size of the tensor.

        Args:
            dim (int, optional): The dimension to return the size of. Defaults to None.

        Returns:
            Union[int, Sequence[int]]: The size of the tensor.
        """
        if dim is None:
            return torch.Size(self.shape)
        return self.shape[dim]

    def __add__(self, other) -> "Tensor":
        """
        Add two tensors element-wise.

        Args:
            other (Tensor): The tensor to be added.

        Returns:
            Tensor: The result of the addition.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([self, other], "eltwise_add")

    def __sub__(self, other) -> "Tensor":
        """
        Subtract two tensors element-wise.

        Args:
            other (Tensor): The tensor to be subtracted.

        Returns:
            Tensor: The result of the subtraction.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([self, -other], "eltwise_add")

    def __mul__(self, other) -> "Tensor":
        """
        Multiply two tensors element-wise.

        Args:
            other (Tensor): The tensor to be multiplied.

        Returns:
            Tensor: The result of the multiplication.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([self, other], "eltwise_mul")

    def __truediv__(self, other) -> "Tensor":
        """
        Divide two tensors element-wise.

        Args:
            other (Tensor): The tensor to be divided.

        Returns:
            Tensor: The result of the division.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([self, other], "eltwise_div")

    def __radd__(self, other) -> "Tensor":
        """
        Add two tensors element-wise.

        Args:
            other (Tensor): The tensor to be added.

        Returns:
            Tensor: The result of the addition.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([other, self], "eltwise_add")

    def __rsub__(self, other) -> "Tensor":
        """
        Subtract two tensors element-wise.

        Args:
            other (Tensor): The tensor to be subtracted.

        Returns:
            Tensor: The result of the subtraction.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([other, -self], "eltwise_add")

    def __rmul__(self, other) -> "Tensor":
        """
        Multiply two tensors element-wise.

        Args:
            other (Tensor): The tensor to be multiplied.

        Returns:
            Tensor: The result of the multiplication.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([other, self], "eltwise_mul")

    def __rtruediv__(self, other) -> "Tensor":
        """
        Divide two tensors element-wise.

        Args:
            other (Tensor): The tensor to be divided.

        Returns:
            Tensor: The result of the division.
        """
        if isinstance(other, (int, float)):
            other = self.factory.constant(
                torch.tensor([other], dtype=self.dtype.torch_dtype)
            )
        return generate_op([other, self], "eltwise_div")

    def __neg__(self) -> "Tensor":
        """
        Negate the tensor.

        Returns:
            Tensor: The negated tensor.
        """
        return generate_op([self], "negative")

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor({self.shape}, {self.dtype})"

    def __str__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor({self.shape}, {self.dtype})"

    def __len__(self) -> int:
        """
        Return the total number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return np.product(self.shape)

    def __getitem__(self, key) -> "Tensor":
        """
        Return a slice of the tensor.

        Args:
            key: The slice key.

        Raises:
            ValueError: If the slice key is invalid.

        Returns:
            Tensor: The sliced tensor.
        """
        shape_len = len(self.shape)

        begin, end, stride = [], [], []
        if isinstance(key, slice):
            key = (key,)
        if not isinstance(key, tuple):
            raise ValueError(
                f"Invalid slice key: must be a tuple instead of {type(key)}"
            )

        if any(k is Ellipsis for k in key):
            # if ellispis is at the start
            if key[0] is Ellipsis:
                key = tuple([slice(None)] * (shape_len - len(key) + 1)) + key[1:]
            # if ellispis is at the end
            if key[-1] is Ellipsis:
                key = key[:-1] + tuple([slice(None)] * (shape_len - len(key) + 1))
            # if ellispis is in the middle
            if any(k is Ellipsis for k in key):
                raise ValueError("Ellipsis must be at the start or end of the slice")

        if len(key) != shape_len or len(key) < 1:
            raise ValueError(f"Invalid slice key: {key}")

        def get_index(idx: int, shape: int) -> int:
            """
            Get the index of the slice.

            Args:
                idx (int): The index of the slice.
                shape (int): The shape of the tensor.

            Raises:
                IndexError: If the index is out of bounds.

            Returns:
                int: The index of the slice.
            """
            if idx < 0:
                idx += shape
            if idx < 0 or idx > shape:
                raise IndexError(f"Index {idx} out of bounds for shape {shape}")
            return idx

        for i, k in enumerate(key):
            if isinstance(k, slice):
                begin.append(get_index(k.start or 0, self.shape[i]))
                end.append(get_index(k.stop or self.shape[i], self.shape[i]))
                stride.append(k.step or 1)
            elif k is None:
                begin.append(0)
                end.append(self.shape[i])
                stride.append(1)
            else:
                begin.append(k)
                end.append(k + 1)
                stride.append(1)

        if any(s <= 0 for s in stride):
            raise ValueError("Stride must be positive")

        return generate_op([self], "slice", begin, end, stride)

    @property
    def T(self) -> "Tensor":
        """
        Return the transpose of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        input_order = list(range(len(self.shape)))
        input_order[-1], input_order[-2] = input_order[-2], input_order[-1]
        return generate_op([self], "transpose", input_order)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.

        Returns:
            Tensor: The transposed tensor.
        """
        input_order = list(range(len(self.shape)))
        input_order[dim0], input_order[dim1] = input_order[dim1], input_order[dim0]

        return generate_op([self], "transpose", input_order)

    def permute(self, *input_order: int) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            input_order (Sequence[int]): The order of the dimensions in the transposed tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        return generate_op([self], "transpose", input_order)

    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            shape (Union[int, Sequence[int]]): The new shape of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]  # type: ignore
        return generate_op([self], "reshape", shape)

    def view(self, *shape: Union[Sequence[int], int]) -> "Tensor":
        """
        Return the transpose of the tensor.

        Args:
            shape (Union[Sequence[int], int]): The new shape of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]  # type: ignore

        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1) -> "Tensor":
        """
        Flatten the tensor.

        Args:
            start_dim (int): The first dim to flatten. Defaults to 0.
            end_dim (int): The last dim to flatten. Defaults to -1.

        Returns:
            Tensor: The flattened tensor.
        """
        if end_dim < 0:
            end_dim = len(self.shape) + end_dim + 1

        flattened_dim = self.shape[start_dim:end_dim]
        size = int(np.prod(flattened_dim))
        new_shape = list(self.shape[:start_dim]) + [size] + list(self.shape[end_dim:])

        return self.reshape(*new_shape)

    def squeeze(self) -> "Tensor":
        """
        Remove dimensions of size 1 from the tensor.

        Returns:
            Tensor: The squeezed tensor.
        """
        return generate_op([self], "squeeze")

    def unsqueeze(self, axis) -> "Tensor":
        """
        Add a dimension of size 1 to the tensor.

        Args:
            axis (int): The axis along which to add the dimension.

        Returns:
            Tensor: The unsqueezed tensor.
        """
        return generate_op([self], "unsqueeze", axis)

    def __matmul__(self, other) -> "Tensor":
        """
        Perform matrix multiplication between two tensors.

        Args:
            other (Tensor): The tensor to be multiplied.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        return generate_op([self, other], "matmul", False, False)

    def acos(self) -> "Tensor":
        """
        Apply the acos function to the tensor.

        Returns:
            Tensor: The result of applying the acos function.
        """
        return torch.acos(self)

    def asin(self) -> "Tensor":
        """
        Apply the asin function to the tensor.

        Returns:
            Tensor: The result of applying the asin function.
        """
        return torch.asin(self)

    def atan(self) -> "Tensor":
        """
        Apply the atan function to the tensor.

        Returns:
            Tensor: The result of applying the atan function.
        """
        return torch.atan(self)

    def acosh(self) -> "Tensor":
        """
        Apply the acosh function to the tensor.

        Returns:
            Tensor: The result of applying the acosh function.
        """
        return torch.acosh(self)

    def asinh(self) -> "Tensor":
        """
        Apply the asinh function to the tensor.

        Returns:
            Tensor: The result of applying the asinh function.
        """
        return torch.asinh(self)

    def atanh(self) -> "Tensor":
        """
        Apply the atanh function to the tensor.

        Returns:
            Tensor: The result of applying the atanh function.
        """
        return torch.atanh(self)

    def cosh(self) -> "Tensor":
        """
        Apply the cosh function to the tensor.

        Returns:
            Tensor: The result of applying the cosh function.
        """
        return torch.cosh(self)

    def sinh(self) -> "Tensor":
        """
        Apply the sinh function to the tensor.

        Returns:
            Tensor: The result of applying the sinh function.
        """
        return torch.sinh(self)

    def tanh(self) -> "Tensor":
        """
        Apply the tanh function to the tensor.

        Returns:
            Tensor: The result of applying the tanh function.
        """
        return torch.tanh(self)

    def cos(self) -> "Tensor":
        """
        Apply the cos function to the tensor.

        Returns:
            Tensor: The result of applying the cos function.
        """
        return torch.cos(self)

    def sin(self) -> "Tensor":
        """
        Apply the sin function to the tensor.

        Returns:
            Tensor: The result of applying the sin function.
        """
        return torch.sin(self)

    def tan(self) -> "Tensor":
        """
        Apply the tan function to the tensor.

        Returns:
            Tensor: The result of applying the tan function.
        """
        return torch.tan(self)

    def ceiling(self) -> "Tensor":
        """
        Apply the ceiling function to the tensor.

        Returns:
            Tensor: The result of applying the ceiling function.
        """
        return generate_op([self], "ceiling")

    def clamp(self, min=None, max=None) -> "Tensor":
        """
        Apply the clamp function to the tensor.

        Args:
            min (int, float): The lower-bound of the range to be clamped
            max (int, float): The upper-bound of the range to be clamped

        Returns:
            Tensor: The result of applying the ceil function.
        """
        return torch.clamp(self, min=min, max=max)

    def elu(self, alpha: float = 1.0) -> "Tensor":
        """
        Apply the elu function to the tensor.

        Args:
            alpha (float): The alpha value. Defaults to 1.0.

        Returns:
            Tensor: The result of applying the elu function.
        """
        return generate_op([self], "elu", alpha)

    def erf(self) -> "Tensor":
        """
        Apply the erf function to the tensor.

        Returns:
            Tensor: The result of applying the erf function.
        """
        return torch.erf(self)

    def exp(self) -> "Tensor":
        """
        Apply the exp function to the tensor.

        Returns:
            Tensor: The result of applying the exp function.
        """
        return torch.exp(self)

    def floor(self) -> "Tensor":
        """
        Apply the floor function to the tensor.

        Returns:
            Tensor: The result of applying the floor function.
        """
        return torch.floor(self)

    def grn(self, bias: float = 1e-12) -> "Tensor":
        """
        Apply the grn function to the tensor.

        Args:
            bias (float): The bias value. Defaults to 1e-12.

        Returns:
            Tensor: The result of applying the grn function.
        """
        return generate_op([self], "grn", bias)

    def hsigmoid(self) -> "Tensor":
        """
        Apply the hsigmoid function to the tensor.

        Returns:
            Tensor: The result of applying the hsigmoid function.
        """
        return generate_op([self], "hsigmoid")

    def hswish(self) -> "Tensor":
        """
        Apply the hswish function to the tensor.

        Returns:
            Tensor: The result of applying the hswish function.
        """
        return generate_op([self], "hswish")

    def log(self) -> "Tensor":
        """
        Apply the log function to the tensor.

        Returns:
            Tensor: The result of applying the log function.
        """
        return torch.log(self)

    def mish(self) -> "Tensor":
        """
        Apply the mish function to the tensor.

        Returns:
            Tensor: The result of applying the mish function.
        """
        return generate_op([self], "mish")

    def relu(self) -> "Tensor":
        """
        Apply the relu function to the tensor.

        Returns:
            Tensor: The result of applying the relu function.
        """
        return generate_op([self], "relu")

    def round(self) -> "Tensor":
        """
        Apply the round function to the tensor.

        Returns:
            Tensor: The result of applying the round function.
        """
        return torch.round(self)

    def sigmoid(self) -> "Tensor":
        """
        Apply the sigmoid function to the tensor.

        Returns:
            Tensor: The result of applying the sigmoid function.
        """
        return generate_op([self], "sigmoid")

    def sign(self) -> "Tensor":
        """
        Apply the sign function to the tensor.

        Returns:
            Tensor: The result of applying the sign function.
        """
        return torch.sign(self)

    def softmax(self, dim) -> "Tensor":
        """
        Apply the softmax function to the tensor.

        Args:
            dim (int): The dimension to apply softmax.

        Returns:
            Tensor: The result of applying the softmax function.
        """
        return torch.nn.functional.softmax(self, dim=dim)

    def softplus(self) -> "Tensor":
        """
        Apply the softplus function to the tensor.

        Returns:
            Tensor: The result of applying the softplus function.
        """
        return generate_op([self], "softplus")

    def sqrt(self) -> "Tensor":
        """
        Apply the sqrt function to the tensor.

        Returns:
            Tensor: The result of applying the sqrt function.
        """
        return torch.sqrt(self)

    def max(
        self, dim: Optional[int] = None, keep_dims: Optional[bool] = False
    ) -> "Tensor":
        """
        Return the reduced max tensor.

        Args:
            dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
            keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.

        Returns:
            Tensor: The result of max reducing operation.
        """
        return generate_op(self, "reduce_max", reduction_axes=dim, keep_dims=keep_dims)

    def mean(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Tensor":
        """
        Return the reduced mean tensor.

        Args:
            dim (Optional[Union[int, Sequence[int]]], optional): The dim(s) to reduce. Default is None, and all dimensions are reduced.
            keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
            dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

        Returns:
            Tensor: The result of mean reducing operation.
        """
        mean = generate_op(self, "reduce_mean", reduction_axes=dim, keep_dims=keep_dims)
        if dtype:
            mean = mean.to(dtype)
        return mean

    def min(
        self,
        dim: Optional[int] = None,
        keep_dims: Optional[bool] = False,
    ) -> "Tensor":
        """
        Return the reduced min tensor.

        Args:
            dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
            keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.

        Returns:
            Tensor: The result of min reducing operation.
        """
        return generate_op(self, "reduce_min", reduction_axes=dim, keep_dims=keep_dims)

    def prod(
        self,
        dim: Optional[int] = None,
        keep_dims: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Tensor":
        """
        Return the reduced product tensor.

        Args:
            dim (Optional[int], optional): The dim to reduce. Default is None, and all dimensions are reduced.
            keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
            dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

        Returns:
            Tensor: The result of product reducing operation.
        """
        prod = generate_op(self, "reduce_prod", reduction_axes=dim, keep_dims=keep_dims)
        if dtype:
            prod = prod.to(dtype)
        return prod

    def sum(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        keep_dims: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Tensor":
        """
        Return the reduced sum tensor.

        Args:
            dim (Optional[Union[int, Sequence[int]]], optional): The dim(s) to reduce. Default is None, and all dimensions are reduced.
            keep_dims (Optional[bool], optional): If set to 1 it holds axes that are used for reduction. Defaults to False.
            dtype (Optional[torch.dtype], optional): The data type. Defaults to None.

        Returns:
            Tensor: The result of sum reducing operation.
        """
        sum = generate_op(self, "reduce_sum", reduction_axes=dim, keep_dims=keep_dims)
        if dtype:
            sum = sum.to(dtype)
        return sum

    def chunk(
        self,
        chunks: int,
        dim: int = 0,
    ) -> Union["Tensor", list]:
        """
        Return the list of tensor chunks.

        Args:
            chunks (int): The number of chunks to return.
            dim (int): The dimension along which to split the tensor. Default is 0.

        Returns:
            Union["Tensor", list]: The resulting list of split tensors or a single tensor.

        Raises:
            ValueError: The input chunks value is not valid.
        """
        if chunks <= 0:
            raise ValueError("The input chunks value is not valid.")
        if chunks == 1:
            return self
        tensors = []
        remainder = self.shape[dim] % chunks
        chunk_size = self.shape[dim] // chunks + (1 if remainder > 0 else 0)
        num_dims = self.dim()

        start_idx = 0
        for _ in range(chunks):
            indexes = [slice(None)] * num_dims
            end_idx = start_idx + chunk_size
            end_idx = end_idx if end_idx < self.shape[dim] else self.shape[dim]
            indexes[dim] = slice(start_idx, end_idx)
            tensors.append(self.__getitem__(tuple(indexes)))
            start_idx = end_idx
        return tensors

    def to(self, dtype: NPUDtype) -> "Tensor":
        """
        Convert the tensor to the specified data type.

        Args:
            dtype (NPUDtype): The data type to convert the tensor to.

        Returns:
            Tensor: The converted tensor.
        """
        return generate_op([self], "to", dtype)

    def type(self, dtype: NPUDtype) -> "Tensor":
        """
        Convert the tensor to the specified data type.

        Args:
            dtype (NPUDtype): The data type to convert the tensor to.

        Returns:
            Tensor: The converted tensor.
        """
        return self.to(dtype)

    @classmethod
    def __torch_function__(
        cls: Any,
        func: Any,
        types: Any,
        args: Sequence[Any] = (),
        kwargs: Optional[MutableMapping[Any, Any]] = None,
    ) -> Any:
        """Python function to override torch functions for Tensor class.

        Args:
            func (Any): the function to override.
            types (Any): the types of the arguments.
            args (Sequence[Any], optional): the arguments. Defaults to ().
            kwargs (Optional[MutableMapping[Any, Any]], optional): the keyword arguments. Defaults to None.

        Returns:
            Any: the result of the function.
        """
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Tensor)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


HANDLED_FUNCTIONS: MutableMapping[Any, Any] = {}


def implements(torch_function: Any) -> Any:
    """Implement a decorator to override torch functions for Tensor class.

    Args:
        torch_function (Any): the function to override.

    Returns:
        Any: the result of the function.
    """

    def decorator(func: Any) -> Any:
        """Implement a decorator to override torch functions for Tensor class.

        Args:
            func (Any): the function to override.

        Returns:
            Any: the result of the function.
        """
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def generate_op(
    tensors: Union[Sequence[Union[Tensor, torch.Tensor]], Union[Tensor, torch.Tensor]],
    op: str,
    *args: Any,
    **kwargs: Any,
) -> "Tensor":
    """
    Generate a new tensor by applying the specified operation to a sequence of tensors.

    Args:
        tensors (Union[Sequence[Union[Tensor, torch.Tensor]], Union[Tensor, torch.Tensor]]): A sequence or a single tensor.
        op (str): The name of the operation to apply.
        args (Any): Variable length argument list.
        kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor generated by applying the operation to the input tensors.

    Raises:
        ValueError: If the tensors are not from the same factory.

    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    # Check that all tensors are from the same factory
    if (
        not len({tensor.factory for tensor in tensors if isinstance(tensor, Tensor)})
        == 1
    ):
        raise ValueError("All tensors must be from the same factory")

    # Get the first factory from the tensors
    factory = [t for t in tensors if isinstance(t, Tensor)][0].factory

    # Replace the tensors that are not from the factory with constant tensors if they are coming from pytorch
    tensors = [
        tensor if isinstance(tensor, Tensor) else factory.constant(tensor)
        for tensor in tensors
    ]

    # Create the operation
    return factory.__getattribute__(op)(*tensors, *args, **kwargs)

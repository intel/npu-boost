#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.ops import get_supported_ops
import numpy as np
import warnings
import ctypes
import sys
import os

handler = ctypes.POINTER(ctypes.c_char)
c_fp16_array = np.ctypeslib.ndpointer(dtype=np.float16, ndim=2, flags="C_CONTIGUOUS")
c_fp32_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
c_i8_array = np.ctypeslib.ndpointer(dtype=np.int8, ndim=2, flags="C_CONTIGUOUS")
c_u8_array = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")
c_u32_array = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")


def load_library() -> ctypes.CDLL:
    """Load the Intel® NPU Acceleration Library runtime library.

    Raises:
        RuntimeError: an error is raised if the platform is not supported. Currently supported platforms are WIndows and Linux

    Returns:
        ctypes.CDLL: The loaded dynamic library
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if "openvino" in sys.modules:
        warnings.warn(
            "OpenVINO library is already loaded. It might interfere with NPU acceleration library if it uses an old version.",
            stacklevel=2,
        )

    external_path = os.path.join(path, "..", "external")
    sys.path.insert(0, external_path)

    if sys.platform == "win32":
        dll_path = os.path.join(path, "..", "lib", "Release")
        os.environ["OPENVINO_LIB_PATHS"] = dll_path
        os.add_dll_directory(os.path.abspath(dll_path))
        # Load DLL into memory.
        lib = ctypes.WinDLL(
            os.path.join(dll_path, "intel_npu_acceleration_library.dll")
        )  # , winmode=0)
    elif sys.platform == "linux":
        dll_path = os.path.join(path, "..", "lib")
        sys.path.append(dll_path)
        # In Linux it is required to explicitly load openvino lib
        _ = ctypes.CDLL(os.path.join(dll_path, "libopenvino.so"))
        lib = ctypes.CDLL(
            os.path.join(dll_path, "libintel_npu_acceleration_library.so")
        )
    else:
        raise RuntimeError(
            f"Platform {sys.platform} is not supported for intel-npu-acceleration-library library"
        )

    return lib


def init_common(lib: ctypes.CDLL):
    """Initialize common runtime bindings.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.saveModel.argtypes = [handler, ctypes.c_char_p]
    lib.saveCompiledModel.argtypes = [handler, ctypes.c_char_p]

    # Set input activations
    lib.set_activation.argtypes = [handler, ctypes.c_void_p, ctypes.c_int]

    # Set outputs activations
    lib.set_output.argtypes = [handler, ctypes.c_void_p, ctypes.c_int]

    # Run a linar layer
    lib.run.argtypes = [handler]
    lib.run.restype = ctypes.c_float

    # Common destructor
    lib.destroyNNFactory.argtypes = [handler]

    lib.isNPUAvailable.restype = ctypes.c_bool
    lib.getNPUDriverVersion.restype = ctypes.c_int32

    lib.compressToI4.argtypes = [c_i8_array, c_u8_array, ctypes.c_int]

    # Remote tensors
    lib.to_npu.argtypes = [ctypes.c_int, c_u32_array, ctypes.c_char_p, ctypes.c_void_p]
    lib.to_npu.restype = handler

    lib.remote_tensor_data.argtypes = [handler]
    lib.remote_tensor_data.restype = ctypes.c_void_p

    lib.del_remote_tensor.argtypes = [handler]


def init_network_factory(lib: ctypes.CDLL):
    """Initialize Netowrk factory bindings.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.createNNFactory.argtypes = [
        ctypes.c_char_p,
        ctypes.c_bool,
    ]
    lib.createNNFactory.restype = handler

    lib.setNNFactoryWeights.argtypes = [handler, handler]

    lib.op_shape_size.argtypes = [handler]
    lib.op_shape_size.restype = ctypes.c_int

    lib.op_shape.argtypes = [handler, ctypes.c_int]
    lib.op_shape.restype = ctypes.c_int

    lib.op_dtype.argtypes = [handler]
    lib.op_dtype.restype = ctypes.c_int

    lib.parameter.argtypes = [handler, ctypes.c_int, c_u32_array, ctypes.c_char_p]
    lib.parameter.restype = handler

    lib.to.argtypes = [handler, handler, ctypes.c_char_p]
    lib.to.restype = handler

    lib.constant.argtypes = [
        handler,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.constant.restype = handler

    lib.slice.argtypes = [
        handler,
        handler,
        handler,
        handler,
        handler,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
    ]
    lib.slice.restype = handler

    lib.compile.argtypes = [handler]
    lib.compile.restype = handler

    lib.get_output_tensor_shape_size.argtypes = [handler, ctypes.c_int]
    lib.get_output_tensor_shape_size.restype = ctypes.c_int

    lib.get_output_tensor_shape.argtypes = [handler, ctypes.c_int, ctypes.c_int]
    lib.get_output_tensor_shape.restype = ctypes.c_int

    lib.linear.argtypes = [
        handler,
        handler,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.linear.restype = handler

    lib.convolution.argtypes = [
        handler,
        handler,
        handler,
        handler,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.convolution.restype = handler

    lib.avg_pooling.argtypes = [
        handler,
        handler,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_bool,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.avg_pooling.restype = handler

    lib.max_pooling.argtypes = [
        handler,
        handler,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        c_u32_array,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.max_pooling.restype = handler

    for op in get_supported_ops():
        fn = getattr(lib, op.name)
        fn.argtypes = [handler] * (op.inputs + 1) + list(op.parameters)
        fn.restype = handler


def init_parameters(lib: ctypes.CDLL):
    """Initialize Netowrk factory parameters.

    Args:
        lib (ctypes.CDLL): Intel® NPU Acceleration Library runtime library
    """
    lib.createParameters.argtypes = []
    lib.createParameters.restype = handler

    lib.destroyParameters.argtypes = [handler]

    lib.addFloatParameter.argtypes = [handler, c_fp16_array, ctypes.c_int, ctypes.c_int]
    lib.addIntParameter.argtypes = [
        handler,
        c_i8_array,
        c_fp16_array,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.addInt4Parameter.argtypes = [
        handler,
        c_u8_array,
        c_fp16_array,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.addIntParameterConversion.argtypes = [
        handler,
        c_i8_array,
        c_fp32_array,
        ctypes.c_int,
        ctypes.c_int,
    ]


def initialize_bindings() -> ctypes.CDLL:
    """Load the Intel® NPU Acceleration Library runtime library, and initialize all c++ <-> python bindings.

    Returns:
        ctypes.CDLL: Initialize matmul bindings
    """
    lib = load_library()

    init_common(lib)
    init_network_factory(lib)
    init_parameters(lib)

    return lib


lib = initialize_bindings()

#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.backend.tensor import Tensor
from intel_npu_acceleration_library.backend import NNFactory
from intel_npu_acceleration_library.dtypes import (
    float16,
    float32,
    float64,
    int4,
    int8,
    int16,
    int32,
    int64,
)
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("shape", [[1, 128, 13, 13], [12, 231]])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        float16,
        float32,
        float64,
        int4,
        int8,
        int16,
        int32,
        int64,
    ],
)
def test_tensor_creation(shape, dtype):
    model = NNFactory()
    tensor = model.parameter(shape, dtype)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == shape
    assert tensor.dtype == dtype


def test_model_creation():
    model = NNFactory()
    t1 = model.parameter([1, 128, 32, 64], float16)
    assert t1.shape == [1, 128, 32, 64]
    assert t1.dtype == float16
    t2 = model.parameter([128 * 32 * 64], int8)
    assert t2.shape == [128 * 32 * 64]
    assert t2.dtype == int8

    t2 = t2.to(float16)

    assert t2.dtype == float16

    t2 = t2.reshape([128, 64, 32])

    assert t2.shape == [128, 64, 32]

    t2 = t2.unsqueeze(0)

    assert t2.shape == [1, 128, 64, 32]

    t2 = t2.T

    assert t2.shape == [1, 128, 32, 64]

    sum = t1 + t2

    model.compile(sum.node)

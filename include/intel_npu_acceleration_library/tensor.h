//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/common.h"

namespace intel_npu_acceleration_library {

/**
 * @brief Class representing a NPU tensor
 *
 */
class Tensor {
private:
    ov::intel_npu::level_zero::ZeroBufferTensor _remote_tensor;
    void* data_ptr;

public:
    /**
     * @brief Construct a new Tensor object
     *
     * @param dtype tensor datatype
     * @param shape tensor shape
     * @param data pointer to tensor data
     * @param tensor_type tensor type. Choices between INPUT, OUTPUT, BINDED
     * @param device target device for the tensor
     */
    Tensor(ov::element::Type_t dtype, ov::Shape shape, void* data,
           ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::INPUT, std::string device = "NPU") {
        if (!_isNPUAvailable(core)) {
            // Cannot create NPU remote tensor... use the same pointer as before
            data_ptr = data;
        } else {
            auto context = core.get_default_context(device).as<ov::intel_npu::level_zero::ZeroContext>();
            _remote_tensor = context.create_l0_host_tensor(dtype, shape, tensor_type);
            data_ptr = _remote_tensor.get();
            std::memcpy(data_ptr, data, _remote_tensor.get_byte_size());
        }
    }

    /**
     * @brief Get the data pointer
     *
     * @return void*
     */
    void* data() {
        return data_ptr;
    }
};

}  // namespace intel_npu_acceleration_library
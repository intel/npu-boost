//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
#define intel_npu_acceleration_library_DLL_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define intel_npu_acceleration_library_DLL_API __declspec(dllexport)
#endif

namespace intel_npu_acceleration_library {

/**
 * @brief OpenVINO core object
 *
 */
ov::Core core;

static constexpr ov::Property<std::string> npu_compiler_type{"NPU_COMPILER_TYPE"};
static constexpr ov::Property<std::string> npu_parameters{"NPU_COMPILATION_MODE_PARAMS"};

/**
 * @brief Return true if the NPU is available on the system, otherwise return false
 *
 * @param core ov::Cor object
 * @return true NPU AI accelerator is available
 * @return false NPU AI accelerator is not available
 */
bool _isNPUAvailable(ov::Core& core) {
    std::vector<std::string> availableDevices = core.get_available_devices();
    return std::find(availableDevices.begin(), availableDevices.end(), "NPU") != availableDevices.end();
}

uint32_t driver_version(ov::Core& core) {
    return static_cast<uint32_t>(core.get_property("NPU", ov::intel_npu::driver_version));
}

ov::element::Type_t dtype_from_string(const std::string& dtype) {
    if (dtype == "int8" || dtype == "i8") {
        return ov::element::Type_t::i8;
    } else if (dtype == "int4" || dtype == "i4") {
        return ov::element::Type_t::i4;
    } else if (dtype == "int16" || dtype == "i16") {
        return ov::element::Type_t::i16;
    } else if (dtype == "int32" || dtype == "i32") {
        return ov::element::Type_t::i32;
    } else if (dtype == "int64" || dtype == "i64") {
        return ov::element::Type_t::i64;
    }
    if (dtype == "float16" || dtype == "half" || dtype == "f16") {
        return ov::element::Type_t::f16;
    }
    if (dtype == "float32" || dtype == "f32") {
        return ov::element::Type_t::f32;
    }
    if (dtype == "float64" || dtype == "f64") {
        return ov::element::Type_t::f64;
    }
    if (dtype == "bfloat16" || dtype == "bf16") {
        return ov::element::Type_t::bf16;
    } else {
        throw std::invalid_argument("Unsupported datatype: " + dtype);
    }
}

}  // namespace intel_npu_acceleration_library

// Define half pointer as uint16_t pointer datatype
#define half_ptr uint16_t*
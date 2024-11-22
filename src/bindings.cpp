//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/nn_factory.h"

extern "C" {

intel_npu_acceleration_library_DLL_API bool isNPUAvailable() {
    return intel_npu_acceleration_library::_isNPUAvailable(intel_npu_acceleration_library::core);
}

intel_npu_acceleration_library_DLL_API uint32_t getNPUDriverVersion() {
    return intel_npu_acceleration_library::driver_version(intel_npu_acceleration_library::core);
}

// ######################## Remote Tensors ########################

intel_npu_acceleration_library_DLL_API intel_npu_acceleration_library::Tensor* to_npu(size_t size,
                                                                                      unsigned int* shape_data,
                                                                                      char* dtype, void* data) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    std::vector<size_t> shape(shape_data, shape_data + size);

    return new intel_npu_acceleration_library::Tensor(ov_dtype, shape, data);
}

intel_npu_acceleration_library_DLL_API void* remote_tensor_data(intel_npu_acceleration_library::Tensor* rt) {
    return rt->data();
}

intel_npu_acceleration_library_DLL_API void del_remote_tensor(intel_npu_acceleration_library::Tensor* rt) {
    delete rt;
}

// ######################## Compression ########################

intel_npu_acceleration_library_DLL_API void compressToI4(const int8_t* src, uint8_t* dst, size_t size) {
    intel_npu_acceleration_library::compressToI4(src, dst, size);
}

// ######################### Parameters #########################

intel_npu_acceleration_library_DLL_API intel_npu_acceleration_library::Parameters* createParameters() {
    return new intel_npu_acceleration_library::Parameters();
}

intel_npu_acceleration_library_DLL_API void destroyParameters(intel_npu_acceleration_library::Parameters* parameters) {
    delete parameters;
}

intel_npu_acceleration_library_DLL_API void addFloatParameter(intel_npu_acceleration_library::Parameters* parameters,
                                                              half_ptr data, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addIntParameter(intel_npu_acceleration_library::Parameters* parameters,
                                                            int8_t* data, half_ptr scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addInt4Parameter(intel_npu_acceleration_library::Parameters* parameters,
                                                             uint8_t* data, half_ptr scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

intel_npu_acceleration_library_DLL_API void addIntParameterConversion(
        intel_npu_acceleration_library::Parameters* parameters, int8_t* data, float* scale, size_t dim0, size_t dim1) {
    parameters->add_parameter(data, scale, intel_npu_acceleration_library::Shape({dim0, dim1}));
}

// ######################### NN Factory #########################

intel_npu_acceleration_library_DLL_API intel_npu_acceleration_library::ModelFactory* createNNFactory(
        char* device, bool profile = false) {
    return new intel_npu_acceleration_library::ModelFactory(std::string(device), profile);
}

intel_npu_acceleration_library_DLL_API void destroyNNFactory(intel_npu_acceleration_library::OVInferenceModel* matmul) {
    delete matmul;
}

intel_npu_acceleration_library_DLL_API void saveModel(intel_npu_acceleration_library::OVInferenceModel* matmul,
                                                      char* path) {
    matmul->saveModel(std::string(path));
}

intel_npu_acceleration_library_DLL_API void saveCompiledModel(intel_npu_acceleration_library::OVInferenceModel* matmul,
                                                              char* path) {
    matmul->saveCompiledModel(std::string(path));
}

intel_npu_acceleration_library_DLL_API void setNNFactoryWeights(
        intel_npu_acceleration_library::OVInferenceModel* mm, intel_npu_acceleration_library::Parameters* parameters) {
    mm->wt_thread = std::thread(&intel_npu_acceleration_library::OVInferenceModel::setWeights, mm,
                                parameters->get_parameters());
}

intel_npu_acceleration_library_DLL_API void compile(intel_npu_acceleration_library::ModelFactory* factory) {
    factory->compile();
}

intel_npu_acceleration_library_DLL_API void result(intel_npu_acceleration_library::ModelFactory* factory,
                                                   ov::op::Op* result) {
    factory->result(result);
}

intel_npu_acceleration_library_DLL_API size_t
get_output_tensor_shape_size(intel_npu_acceleration_library::ModelFactory* factory, size_t tensor_idx) {
    ov::Tensor tensor = factory->getOutputTensors(tensor_idx);
    return tensor.get_shape().size();
}

intel_npu_acceleration_library_DLL_API size_t
get_output_tensor_shape(intel_npu_acceleration_library::ModelFactory* factory, size_t tensor_idx, size_t idx) {
    ov::Tensor tensor = factory->getOutputTensors(tensor_idx);
    return tensor.get_shape()[idx];
}

intel_npu_acceleration_library_DLL_API void set_activation(intel_npu_acceleration_library::OVInferenceModel* mm,
                                                           void* X, size_t idx) {
    mm->setInputTensor(X, idx);
}

intel_npu_acceleration_library_DLL_API void set_output(intel_npu_acceleration_library::OVInferenceModel* mm, void* Out,
                                                       size_t idx) {
    mm->setOutputTensor(Out, idx);
}

intel_npu_acceleration_library_DLL_API float run(intel_npu_acceleration_library::OVInferenceModel* mm) {
    auto start = std::chrono::system_clock::now();

    mm->run();

    auto stop = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return static_cast<float>(elapsed.count()) / static_cast<float>(1000.0);
}

// ######################### NN Factory ops #########################

intel_npu_acceleration_library_DLL_API size_t op_shape_size(ov::op::Op* in0) {
    return in0->get_output_shape(0).size();
}

intel_npu_acceleration_library_DLL_API size_t op_shape(ov::op::Op* in0, size_t idx) {
    return in0->get_output_shape(0)[idx];
}

intel_npu_acceleration_library_DLL_API size_t op_dtype(ov::op::Op* in0) {
    auto dtype = static_cast<ov::element::Type_t>(in0->get_output_element_type(0));
    return static_cast<size_t>(dtype);
}

// ######################### NN Factory layers #########################

intel_npu_acceleration_library_DLL_API ov::op::Op* parameter(intel_npu_acceleration_library::ModelFactory* factory,
                                                             size_t size, unsigned int* data, char* dtype) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    std::vector<size_t> shape(data, data + size);
    return factory->parameter(shape, ov_dtype);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* constant(intel_npu_acceleration_library::ModelFactory* factory,
                                                            size_t size, unsigned int* data, char* dtype, void* dst) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    std::vector<size_t> shape(data, data + size);
    return factory->constant(ov_dtype, shape, dst);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* matmul(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0, ov::op::Op* in1, bool trA, bool trB) {
    return factory->matmul(in0, in1, trA, trB);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_add(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_add(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_mul(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_mul(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* eltwise_div(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* in1) {
    return factory->eltwise_div(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* abs_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->abs(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* acos_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->acos(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* asin_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->asin(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* atan_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->atan(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* ceiling(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->ceiling(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* clamp(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* in0, float min, float max) {
    return factory->clamp(in0, min, max);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* cos_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->cos(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* cosh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->cosh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* erf_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->erf(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* elu(intel_npu_acceleration_library::ModelFactory* factory,
                                                       ov::op::Op* in0, float alpha) {
    return factory->elu(in0, alpha);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* floor_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->floor(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* grn(intel_npu_acceleration_library::ModelFactory* factory,
                                                       ov::op::Op* in0, float bias) {
    return factory->grn(in0, bias);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* exp_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->exp(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* gelu(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->gelu(in0, ov::op::GeluApproximationMode::TANH);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* gelu_erf(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->gelu(in0, ov::op::GeluApproximationMode::ERF);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* log_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->log(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* negative(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->negative(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* relu(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->relu(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* prelu(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* in0, ov::op::Op* in1) {
    return factory->prelu(in0, in1);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sigmoid(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->sigmoid(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sign(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->sign(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sin_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->sin(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sinh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->sinh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* sqrt_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->sqrt(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* tan_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0) {
    return factory->tan(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* tanh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->tanh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* acosh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->acosh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* asinh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->asinh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* atanh_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->atanh(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* hswish(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0) {
    return factory->hswish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* mish(intel_npu_acceleration_library::ModelFactory* factory,
                                                        ov::op::Op* in0) {
    return factory->mish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softplus(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->softplus(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* hsigmoid(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->hsigmoid(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* round_act(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* in0) {
    return factory->round(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softsign(intel_npu_acceleration_library::ModelFactory* factory,
                                                            ov::op::Op* in0) {
    return factory->softsign(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* swish(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* in0) {
    return factory->swish(in0);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* softmax(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* in0, int axis) {
    return factory->softmax(in0, axis);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* gather(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* input, ov::op::Op* indices, ov::op::Op* axis,
                                                          const size_t batch_dims) {
    return factory->gather(input, indices, axis, batch_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reshape(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* input, ov::op::Op* shape) {
    return factory->reshape(input, shape);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* slice(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* input, ov::op::Op* begin, ov::op::Op* end,
                                                         ov::op::Op* strides, size_t begin_mask_size,
                                                         unsigned int* begin_mask_ptr, size_t end_mask_size,
                                                         unsigned int* end_mask_ptr) {
    std::vector<int64_t> begin_mask(begin_mask_ptr, begin_mask_ptr + begin_mask_size);
    std::vector<int64_t> end_mask(end_mask_ptr, end_mask_ptr + end_mask_size);

    return factory->slice(input, begin, end, strides, begin_mask, end_mask);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* transpose(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* input, ov::op::Op* input_order) {
    return factory->transpose(input, input_order);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* squeeze(intel_npu_acceleration_library::ModelFactory* factory,
                                                           ov::op::Op* input) {
    return factory->squeeze(input);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* unsqueeze(intel_npu_acceleration_library::ModelFactory* factory,
                                                             ov::op::Op* input, ov::op::Op* axis) {
    return factory->unsqueeze(input, axis);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* concat(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* x1, ov::op::Op* x2, int64_t axis) {
    return factory->concat(x1, x2, axis);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reduce_max(intel_npu_acceleration_library::ModelFactory* factory,
                                                              ov::op::Op* input, ov::op::Op* reduction_axes,
                                                              bool keep_dims) {
    return factory->reduce_max(input, reduction_axes, keep_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reduce_mean(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* input, ov::op::Op* reduction_axes,
                                                               bool keep_dims) {
    return factory->reduce_mean(input, reduction_axes, keep_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reduce_min(intel_npu_acceleration_library::ModelFactory* factory,
                                                              ov::op::Op* input, ov::op::Op* reduction_axes,
                                                              bool keep_dims) {
    return factory->reduce_min(input, reduction_axes, keep_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reduce_prod(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* input, ov::op::Op* reduction_axes,
                                                               bool keep_dims) {
    return factory->reduce_prod(input, reduction_axes, keep_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* reduce_sum(intel_npu_acceleration_library::ModelFactory* factory,
                                                              ov::op::Op* input, ov::op::Op* reduction_axes,
                                                              bool keep_dims) {
    return factory->reduce_sum(input, reduction_axes, keep_dims);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* convert_to_fp16(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0) {
    return factory->convert_to(in0, ov::element::Type_t::f16);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* to(intel_npu_acceleration_library::ModelFactory* factory,
                                                      ov::op::Op* in0, char* dtype) {
    ov::element::Type_t ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(dtype));
    return factory->convert_to(in0, ov_dtype);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* linear(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* in0, size_t dim0, size_t dim1, bool bias,
                                                          char* act_dtype, char* wt_dtype) {
    ov::element::Type_t act_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(act_dtype));
    ov::element::Type_t wt_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(wt_dtype));

    bool quantized = wt_ov_dtype == ov::element::Type_t::i8 || wt_ov_dtype == ov::element::Type_t::i4;

    auto weights = factory->parameter({dim0, dim1}, wt_ov_dtype);
    if (quantized) {
        weights = factory->convert_to(weights, act_ov_dtype);
        auto scale = factory->constant<double>(act_ov_dtype, std::vector<size_t>({1, 1}), sqrt(1.0 / dim1));
        in0 = factory->eltwise_mul(in0, scale);
    }

    auto mm = factory->matmul(in0, weights);

    if (quantized) {
        auto scale = factory->parameter({1, dim0}, act_ov_dtype);
        mm = factory->eltwise_mul(mm, scale);
    }

    if (bias) {
        auto bias = factory->parameter({1, dim0}, act_ov_dtype);
        return factory->eltwise_add(mm, bias);
    }
    return mm;
}

intel_npu_acceleration_library_DLL_API ov::op::Op* convolution(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, ov::op::Op* weights, ov::op::Op* bias,
                                                               size_t strides_size, unsigned int* strides_data,
                                                               size_t pad_begins_size, unsigned int* pad_begins_data,
                                                               size_t pad_ends_size, unsigned int* pad_ends_data,
                                                               size_t dilations_size, unsigned int* dilations_data,
                                                               size_t groups, char* act_dtype) {
    ov::element::Type_t act_ov_dtype = intel_npu_acceleration_library::dtype_from_string(std::string(act_dtype));

    // Create vectors from the input data
    std::vector<size_t> strides(strides_data, strides_data + strides_size);
    std::vector<size_t> pad_begins(pad_begins_data, pad_begins_data + pad_begins_size);
    std::vector<size_t> pad_ends(pad_ends_data, pad_ends_data + pad_ends_size);
    std::vector<size_t> dilations(dilations_data, dilations_data + dilations_size);

    auto weight_shape = weights->get_output_shape(0);
    auto wt_ov_dtype = static_cast<ov::element::Type_t>(weights->get_output_element_type(0));

    bool quantized = wt_ov_dtype == ov::element::Type_t::i8 || wt_ov_dtype == ov::element::Type_t::i4;

    if (quantized) {
        weights = factory->convert_to(weights, act_ov_dtype);
        auto scale =
                factory->constant<double>(act_ov_dtype, std::vector<size_t>({1, 1, 1, 1}), sqrt(1.0 / weight_shape[1]));
        in0 = factory->eltwise_mul(in0, scale);
    }

    auto mm = factory->convolution(in0, weights, strides, pad_begins, pad_ends, dilations, groups);

    if (quantized) {
        auto scale = factory->parameter({1, weight_shape[0], 1, 1}, act_ov_dtype);
        mm = factory->eltwise_mul(mm, scale);
    }

    if (bias) {
        return factory->eltwise_add(mm, bias);
    }
    return mm;
}

intel_npu_acceleration_library_DLL_API ov::op::Op* avg_pooling(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* in0, size_t strides_size,
                                                               unsigned int* strides_data, size_t pad_begins_size,
                                                               unsigned int* pad_begins_data, size_t pad_ends_size,
                                                               unsigned int* pad_ends_data, size_t kernel_size,
                                                               unsigned int* kernel_data, bool exclude_pad,
                                                               int rounding_type, int auto_pad) {
    // Create vectors from the input data
    std::vector<size_t> strides(strides_data, strides_data + strides_size);
    std::vector<size_t> pad_begins(pad_begins_data, pad_begins_data + pad_begins_size);
    std::vector<size_t> pad_ends(pad_ends_data, pad_ends_data + pad_ends_size);
    std::vector<size_t> kernel(kernel_data, kernel_data + kernel_size);

    return factory->average_pooling(in0, strides, pad_begins, pad_ends, kernel, exclude_pad,
                                    static_cast<ov::op::RoundingType>(rounding_type),
                                    static_cast<ov::op::PadType>(auto_pad));
}

intel_npu_acceleration_library_DLL_API ov::op::Op* adaptive_avg_pool(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0, ov::op::Op* shape) {
    return factory->adaptive_average_pool(in0, shape);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* max_pooling(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0, size_t strides_size,
        unsigned int* strides_data, size_t pad_begins_size, unsigned int* pad_begins_data, size_t pad_ends_size,
        unsigned int* pad_ends_data, size_t kernel_size, unsigned int* kernel_data, int rounding_type, int auto_pad) {
    // Create vectors from the input data
    std::vector<size_t> strides(strides_data, strides_data + strides_size);
    std::vector<size_t> pad_begins(pad_begins_data, pad_begins_data + pad_begins_size);
    std::vector<size_t> pad_ends(pad_ends_data, pad_ends_data + pad_ends_size);
    std::vector<size_t> kernel(kernel_data, kernel_data + kernel_size);

    return factory->max_pooling(in0, strides, pad_begins, pad_ends, kernel,
                                static_cast<ov::op::RoundingType>(rounding_type),
                                static_cast<ov::op::PadType>(auto_pad));
}

intel_npu_acceleration_library_DLL_API ov::op::Op* adaptive_max_pool(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* in0, ov::op::Op* shape) {
    return factory->adaptive_max_pool(in0, shape);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* scaled_dot_product_attention(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* query, ov::op::Op* key, ov::op::Op* value,
        ov::op::Op* attn_mask, bool is_causal) {
    return factory->scaled_dot_product_attention(query, key, value, attn_mask, is_causal);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* scaled_dot_product_attention_simple(
        intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* query, ov::op::Op* key, ov::op::Op* value,
        bool is_causal) {
    return factory->scaled_dot_product_attention(query, key, value, nullptr, is_causal);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* normL2(intel_npu_acceleration_library::ModelFactory* factory,
                                                          ov::op::Op* data, ov::op::Op* axes, float eps) {
    return factory->normL2(data, axes, eps);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* power(intel_npu_acceleration_library::ModelFactory* factory,
                                                         ov::op::Op* x1, ov::op::Op* x2) {
    return factory->power(x1, x2, ov::op::AutoBroadcastType::NUMPY);
}

intel_npu_acceleration_library_DLL_API ov::op::Op* log_softmax(intel_npu_acceleration_library::ModelFactory* factory,
                                                               ov::op::Op* input, int64_t axis) {
    return factory->log_softmax(input, axis);
}
}
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_global_average_pool.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"
#include <functional>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

Status ComputeAveragePool(
    const uint8_t* x,
    float x_scale,
    uint8_t x_zero_point,
    uint8_t* y,
    float y_scale,
    uint8_t y_zero_point,
    int64_t N,
    int64_t C,
    const std::vector<int64_t>& kernel_dims,
    StorageOrder storage_order,
    concurrency::ThreadPool* tp) {
  int64_t kernel_size = std::accumulate(kernel_dims.begin(), kernel_dims.end(), 1LL, std::multiplies<int64_t>());
  if (storage_order == StorageOrder::NCHW) {
    auto worker = [=](std::ptrdiff_t first, std::ptrdiff_t last) {
      const uint8_t* input = (const uint8_t*)(x + (first * kernel_size));
      uint8_t* output = (uint8_t*)(y + first);
      MlasQLinearGlobalAveragePool(input, x_scale, x_zero_point, output, y_scale, y_zero_point, last - first, kernel_size);
    };
    concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(N * C),
                                            {static_cast<double>(kernel_size), 1.0, static_cast<double>(kernel_size)}, worker);
  } else {
    // auto worker = [x, y, C, kernel_size](std::ptrdiff_t first, std::ptrdiff_t last) {
    //   for (; first < last; ++first) {
    //     auto input_matrix = ConstEigenMatrixMapRowMajor<T>(x + (first * C * kernel_size), kernel_size, C);
    //     auto output_matrix = EigenMatrixMapRowMajor<T>(y + (first * C), 1, C);
    //     output_matrix = input_matrix.template cast<TAccumulate>().colwise().mean().template cast<T>();
    //   }
    // };
    // concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(N),
    //                                         {static_cast<double>(kernel_size * C), 1.0 * C, static_cast<double>(kernel_size * C)},
    //                                         worker);
  }
  return Status::OK();
}

Status QLinearGlobalAveragePool::Compute(OpKernelContext* context) const {
  auto tensor_x_scale = context->Input<Tensor>(1);
  auto tensor_x_zero_point = context->Input<Tensor>(2);
  auto tensor_y_scale = context->Input<Tensor>(3);
  auto tensor_y_zero_point = context->Input<Tensor>(4);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_scale),
              "Input x_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_zero_point),
              "input x_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_scale),
              "input y_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_zero_point),
              "input y_zero_point must be a scalar or 1D tensor of size 1 if given");

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto& X = *context->Input<Tensor>(0);
  const auto& x_shape = X.Shape().GetDims();

  ORT_RETURN_IF_NOT(x_shape.size() >= 3, "Input dimension cannot be less than 3.");
  auto first_dim = x_shape.begin() + (storage_order_ == StorageOrder::NCHW ? 2 : 1);
  std::vector<int64_t> kernel_dims{first_dim, first_dim + (x_shape.size() - 2)};
  int64_t N = x_shape[0];
  int64_t C = storage_order_ == StorageOrder::NCHW ? x_shape[1] : x_shape.back();

  std::vector<int64_t> output_dims{N};
  std::vector<int64_t> one_dims(x_shape.size() - 2, 1LL);
  if (storage_order_ == StorageOrder::NCHW) {
    output_dims.push_back(C);
    output_dims.insert(output_dims.end(), one_dims.begin(), one_dims.end());
  } else {
    output_dims.insert(output_dims.end(), one_dims.begin(), one_dims.end());
    output_dims.push_back(C);
  }
  Tensor& Y = *context->Output(0, output_dims);

  const float x_scale = *(tensor_x_scale->Data<float>());
  const float y_scale = *(tensor_y_scale->Data<float>());
  auto dtype = X.GetElementType();
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      std::cout << "====GlobalAveragePool: NxCxImageSize=" << N << "x" << C << "x" << kernel_dims << std::endl;
      return ComputeAveragePool(X.Data<uint8_t>(), x_scale, *(tensor_x_zero_point->Data<uint8_t>()),
                                Y.MutableData<uint8_t>(), y_scale, *(tensor_y_zero_point->Data<uint8_t>()),
                                N, C, kernel_dims, storage_order_, tp);
    default:
      ORT_THROW("Unsupported 'dtype' value: ", dtype);
  }
}

ONNX_OPERATOR_KERNEL_EX(QLinearGlobalAveragePool, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QLinearGlobalAveragePool);

}  // namespace contrib

}  // namespace onnxruntime

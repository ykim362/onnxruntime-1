// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantized_global_average_pool.h"
#include "core/util/math_cpuonly.h"
// #include "core/framework/tensorprotoutils.h"
// #include "core/providers/common.h"
#include "core/platform/threadpool.h"
// #include "core/mlas/inc/mlas.h"
#include <functional>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T, typename TAccumulate>
Status ComputeAveragePool(
    const T* x,
    T* y,
    int64_t N,
    int64_t C,
    const std::vector<int64_t>& kernel_dims,
    StorageOrder storage_order,
    concurrency::ThreadPool* tp) {
  int64_t kernel_size = std::accumulate(kernel_dims.begin(), kernel_dims.end(), 1LL, std::multiplies<int64_t>());
  if (storage_order == StorageOrder::NCHW) {
    auto worker = [x, y, kernel_size](std::ptrdiff_t first, std::ptrdiff_t last) {
      auto input_matrix = ConstEigenMatrixMapRowMajor<T>(x + (first * kernel_size), last - first, kernel_size);
      auto output_matrix = EigenMatrixMapRowMajor<T>(y + first, last - first, 1);
      output_matrix = input_matrix.template cast<TAccumulate>().rowwise().mean().template cast<T>();
    };
    concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(N * C),
                                            {static_cast<double>(kernel_size), 1.0, static_cast<double>(kernel_size)},
                                            worker);
  } else {
    auto worker = [x, y, C, kernel_size](std::ptrdiff_t first, std::ptrdiff_t last) {
      for (; first < last; ++first) {
        auto input_matrix = ConstEigenMatrixMapRowMajor<T>(x + (first * C * kernel_size), kernel_size, C);
        auto output_matrix = EigenMatrixMapRowMajor<T>(y + (first * C), 1, C);
        output_matrix = input_matrix.template cast<TAccumulate>().colwise().mean().template cast<T>();
      }
    };
    concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(N),
                                            {static_cast<double>(kernel_size * C), 1.0 * C, static_cast<double>(kernel_size * C)},
                                            worker);
  }
  return Status::OK();
}

Status QuantizedGlobalAveragePool::Compute(OpKernelContext* context) const {
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

  auto dtype = X.GetElementType();
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return ComputeAveragePool<int8_t, int32_t>(X.Data<int8_t>(), Y.MutableData<int8_t>(), N, C, kernel_dims, storage_order_, tp);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return ComputeAveragePool<uint8_t, uint32_t>(X.Data<uint8_t>(), Y.MutableData<uint8_t>(), N, C, kernel_dims, storage_order_, tp);
    default:
      ORT_THROW("Unsupported 'dtype' value: ", dtype);
  }
}

ONNX_OPERATOR_KERNEL_EX(QuantizedGlobalAveragePool, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QuantizedGlobalAveragePool);

}  // namespace contrib

}  // namespace onnxruntime

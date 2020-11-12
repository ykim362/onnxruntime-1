// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "common.h"
#include "gradient_control.h"
#include "core/profile/context.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(T, T_GRAD)                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                    \
      InPlaceAccumulator,                                                           \
      kMSDomain,                                                                    \
      1,                                                                            \
      T##_##T_GRAD,                                                                 \
      kCudaExecutionProvider,                                                       \
      KernelDefBuilder()                                                            \
          .Alias(0, 0)                            /* Accumulate tensors in-place */ \
          .InputMemoryType<OrtMemTypeCPUInput>(2) /* Keep do_update in CPU */       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()),         \
      InPlaceAccumulator<T, T_GRAD>);

REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(float, float)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(float, MLFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(MLFloat16, MLFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(MLFloat16, float)

template <typename T>
Status ZeroGradient<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& old_gradient = *ctx->Input<Tensor>(0);
  Tensor& zero_gradient = *ctx->Output(0, old_gradient.Shape());

  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  auto input_defs = Node().InputDefs();
  std::cout << "[gradient_control.cc] batch " << tag << ", zero out " << input_defs[0]->Name() << std::endl;

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
      zero_gradient.template MutableData<T>(),
      0,
      zero_gradient.Shape().Size() * sizeof(T)));

  return Status::OK();
}

#define REGISTER_ZERO_GRADIENT_TYPED(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ZeroGradient,                                               \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .Alias(0, 0) /* Zero out gradients in-place */          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", DataTypeImpl::AllTensorTypes()),  \
      ZeroGradient<T>);
REGISTER_ZERO_GRADIENT_TYPED(float)
REGISTER_ZERO_GRADIENT_TYPED(MLFloat16)

template <typename T, typename T_GRAD>
Status InPlaceAccumulator<T, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  const Tensor& left_addee_buffer = *ctx->Input<Tensor>(0);
  const Tensor& right_addee_buffer = *ctx->Input<Tensor>(1);
  const Tensor* do_update_tensor = ctx->Input<Tensor>(2);
  Tensor& accumulation_output = *ctx->Output(0, left_addee_buffer.Shape());

  auto input_defs = Node().InputDefs();

  std::vector<CudaT> l_buffer(left_addee_buffer.Shape().Size());
  std::vector<CudaT_GRAD> r_buffer(right_addee_buffer.Shape().Size());

  cudaMemcpy(l_buffer.data(), reinterpret_cast<const CudaT*>(left_addee_buffer.template Data<T>()), left_addee_buffer.Shape().Size() * sizeof(CudaT), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_buffer.data(), reinterpret_cast<const CudaT_GRAD*>(right_addee_buffer.template Data<T_GRAD>()), right_addee_buffer.Shape().Size() * sizeof(CudaT_GRAD), cudaMemcpyDeviceToHost);

  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  for (int i = 0; i < left_addee_buffer.Shape().Size(); ++i) {
    std::cout << "[gradient_control.cc] batch " << tag << ", " << input_defs[0]->Name() << "[" << i << "]=" << l_buffer[i] << std::endl;
  }

  for (int i = 0; i < right_addee_buffer.Shape().Size(); ++i) {
    std::cout << "[gradient_control.cc] batch " << tag << ", " << input_defs[1]->Name() << "[" << i << "]=" << r_buffer[i] << std::endl;
  }

  if (do_update_tensor) {
    const bool do_update = *(do_update_tensor->template Data<bool>());
    if (!do_update) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T>(left_addee_buffer, accumulation_output));
      return Status::OK();
    }
  }
  InPlaceAccumulatorImpl(
      reinterpret_cast<const CudaT*>(left_addee_buffer.template Data<T>()),
      reinterpret_cast<const CudaT_GRAD*>(right_addee_buffer.template Data<T_GRAD>()),
      reinterpret_cast<CudaT*>(accumulation_output.template MutableData<T>()),
      right_addee_buffer.Shape().Size());
  
  auto output_defs = Node().OutputDefs();
  std::vector<CudaT> buffer(accumulation_output.Shape().Size());
  cudaMemcpy(buffer.data(), reinterpret_cast<const CudaT*>(accumulation_output.template Data<CudaT>()), accumulation_output.Shape().Size() * sizeof(CudaT), cudaMemcpyDeviceToHost);
  for (int i = 0; i < accumulation_output.Shape().Size(); ++i) {
    std::cout << "[gradient_control.cc] batch " << tag << ", " << output_defs[0]->Name() << "[" << i << "]=" << buffer[i] << std::endl;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

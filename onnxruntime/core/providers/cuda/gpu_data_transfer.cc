// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/gpu_data_transfer.h"
#include "cuda_common.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer(const CUDAExecutionProvider* provider) : provider_(provider) {
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));
}

GPUDataTransfer::~GPUDataTransfer() {
  CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::CUDA_PINNED
         || dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, GetStream(exec_queue_id)));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, GetStream(kCudaStreamDefault)));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, GetStream(kCudaStreamDefault)));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(kCudaStreamDefault)));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, GetStream(exec_queue_id)));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, GetStream(kCudaStreamDefault)));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(GetStream(kCudaStreamDefault)));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}
}  // namespace onnxruntime

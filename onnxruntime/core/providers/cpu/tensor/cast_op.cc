// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iomanip>
#include <sstream>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/Default/Half.h"

#if defined(_M_AMD64)
#include "core/mlas/inc/mlas.h"
#endif

#define CAST_STRING_ENABLED
#define CAST_FLOAT16_ENABLED

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

namespace {
template <typename SrcType, typename DstType>
inline void CastData(const Tensor& in, Tensor& out, const TensorShape& shape) {
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<SrcType>(in.Data<SrcType>(), shape_size);
  auto output_vector = EigenVectorMap<DstType>(out.MutableData<DstType>(), shape_size);
  output_vector = in_vector.template cast<DstType>();
}

#ifdef CAST_FLOAT16_ENABLED
template <>
inline void CastData<float, MLFloat16>(const Tensor& in, Tensor& out, const TensorShape& shape) {
  auto out_data = out.MutableData<MLFloat16>();
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<float>(in.Data<float>(), shape_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(out_data)), shape_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

template <>
inline void CastData<MLFloat16, float>(const Tensor& in, Tensor& out, const TensorShape& shape) {
  auto out_data = out.MutableData<float>();
  auto in_data = in.Data<MLFloat16>();
  auto shape_size = shape.Size();
#if defined(_M_AMD64)
  MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
#else
  auto in_vector = ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(in_data)), shape_size);
  auto output_vector = EigenVectorMap<float>(out_data, shape_size);
  output_vector = in_vector.template cast<float>();
#endif
}
#endif

#ifdef CAST_STRING_ENABLED
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToStringData(const Tensor& in, Tensor& out, const TensorShape& shape) {
  // floating point input
  const int64_t len = shape.Size();
  const auto input_data = in.DataAsSpan<SrcType>();
  auto output_data = out.MutableDataAsSpan<std::string>();

  for (int i = 0; i < len; ++i) {
    if (std::isnan(input_data[i])) {
      output_data[i] = "NaN";
    } else if (std::isinf(input_data[i])) {
      if (input_data[i] < std::numeric_limits<SrcType>::lowest()) {
        output_data[i] = "-INF";
      } else {
        output_data[i] = "INF";
      }
    } else {
      std::ostringstream convert;
      if (std::is_floating_point<SrcType>::value) {
        // match numpy default behavior
        convert << std::setprecision(8);
      }

      convert << input_data[i];
      output_data[i] = convert.str();
    }
  }
}

template <typename SrcType>
typename std::enable_if<!std::is_floating_point<SrcType>::value, void>::type
CastToStringData(const Tensor& in, Tensor& out, const TensorShape& shape) {
  const int64_t len = shape.Size();
  const auto input_data = in.DataAsSpan<SrcType>();
  auto output_data = out.MutableDataAsSpan<std::string>();

  for (int i = 0; i < len; ++i) {
    output_data[i] = (std::ostringstream() << input_data[i]).str();
  }
}

template <typename DstType>
void CastFromStringData(const Tensor& in, Tensor& out, const TensorShape& shape) {
  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0);
  if (std::is_same<DstType, float>::value) {
    auto* mutable_data = out.MutableData<float>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stof(in.Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, double>::value) {
    auto* mutable_data = out.MutableData<double>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stod(in.Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int8_t>::value) {
    auto* mutable_data = out.MutableData<int8_t>();
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in.Data<std::string>()[i]);
      mutable_data[i] = static_cast<int8_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint8_t>::value) {
    auto* mutable_data = out.MutableData<uint8_t>();
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in.Data<std::string>()[i]);
      mutable_data[i] = static_cast<uint8_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int16_t>::value) {
    auto* mutable_data = out.MutableData<int16_t>();
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in.Data<std::string>()[i]);
      mutable_data[i] = static_cast<int16_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint16_t>::value) {
    auto* mutable_data = out.MutableData<uint16_t>();
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in.Data<std::string>()[i]);
      mutable_data[i] = static_cast<uint16_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int32_t>::value) {
    auto* mutable_data = out.MutableData<int32_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stol(in.Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint32_t>::value) {
    auto* mutable_data = out.MutableData<uint32_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoul(in.Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int64_t>::value) {
    auto* mutable_data = out.MutableData<int64_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoll(in.Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint64_t>::value) {
    auto* mutable_data = out.MutableData<uint64_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoull(in.Data<std::string>()[i]);
    }
  } else {
    ORT_THROW("Unsupported type in cast op: from String to ", typeid(DstType).name());
  }
}
#endif

}  // namespace

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename TSrc>
  struct SrcDispatcher;

  template <typename TSrc, typename TDest>
  struct Dispatcher;

  template <typename T>
  struct StringDispatcher;

  ONNX_NAMESPACE::TensorProto_DataType to_;
};

const std::vector<MLDataType> castOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>(),
    DataTypeImpl::GetTensorType<int8_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<MLFloat16>(),
    DataTypeImpl::GetTensorType<std::string>()};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    8,
    KernelDefBuilder()
        .TypeConstraint("T1", castOpTypeConstraints)
        .TypeConstraint("T2", castOpTypeConstraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    9,
    KernelDefBuilder()
        .TypeConstraint("T1", castOpTypeConstraints)
        .TypeConstraint("T2", castOpTypeConstraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// default dispatch
template <typename TSrc, typename TDst>
struct Cast::Dispatcher {
  Status operator()(const Tensor& src, Tensor& dst, const TensorShape& shape) {
    CastData<TSrc, TDst>(src, dst, shape);
    return Status::OK();
  }
};

template <typename TSrc>
struct Cast::SrcDispatcher {
  Status operator()(int32_t to, const Tensor& src, Tensor& dst, const TensorShape& shape) {
    utils::MLTypeCallDispatcherRetWithCarriedType<Status, TSrc, Cast::Dispatcher,
                                                  float, double, int8_t, uint8_t, int16_t, uint16_t,
                                                  int32_t, uint32_t, int64_t, uint64_t, bool>
        t_disp(to);

    auto status = t_disp.Invoke(src, dst, shape);

    return Status::OK();
  }
};

#ifdef CAST_STRING_ENABLED
template <typename T>
struct Cast::StringDispatcher {
  Status operator()(bool to_string, const Tensor& src, Tensor& dst, const TensorShape& shape) {
    if (to_string) {
      CastToStringData<T>(src, dst, shape);
    } else {
      CastFromStringData<T>(src, dst, shape);
    }

    return Status::OK();
  }
};
#endif

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));

  auto from = X->GetElementType();
  Status status = Status::OK();

  if (shape.Size() == 0) {
    return status;
  }

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return status;
  }

#ifdef CAST_STRING_ENABLED
  // special case strings
  if (from == ONNX_NAMESPACE::TensorProto_DataType_STRING ||
      to_ == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    bool to_string = to_ == ONNX_NAMESPACE::TensorProto_DataType_STRING;

    utils::MLTypeCallDispatcherRet<Status, StringDispatcher,
                                   float, double, MLFloat16, /*BFloat16,*/
                                   int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
                                   bool>
        t_disp(to_string ? from : to_);

    status = t_disp.Invoke(to_string, *X, *Y, shape);
  } else
#endif
  {
    auto do_cast = [](int32_t from, int32_t to, const Tensor& src, Tensor& dst, const TensorShape& shape) {
      utils::MLTypeCallDispatcherRet<Status, SrcDispatcher,
                                     float, double,  // MLFloat16 is special cased below
                                     int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, bool>
          t_disp(from);

      return t_disp.Invoke(to, src, dst, shape);
    };

#ifdef CAST_FLOAT16_ENABLED
    // MLFloat16  needs special handling
    if (from == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      if (to_ == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        CastData<MLFloat16, float>(*X, *Y, shape);
      } else {
        // need to cast to float first in a temporary buffer
        AllocatorPtr allocator;
        ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
        auto tmp_buffer = IAllocator::MakeUniquePtr<float>(allocator, shape.Size());
        Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, tmp_buffer.get(), allocator->Info());

        CastData<MLFloat16, float>(*X, tmp_tensor, shape);
        status = do_cast(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, to_, tmp_tensor, *Y, shape);
      }
    } else if (to_ == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      if (from == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        CastData<float, MLFloat16>(*X, *Y, shape);
      } else {
        // need to cast to float first in a temporary buffer
        AllocatorPtr allocator;
        ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
        auto tmp_buffer = IAllocator::MakeUniquePtr<float>(allocator, shape.Size());
        Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, tmp_buffer.get(), allocator->Info());

        ORT_RETURN_IF_ERROR(do_cast(from, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, *X, tmp_tensor, shape));
        CastData<float, MLFloat16>(tmp_tensor, *Y, shape);
      }
    } else
#endif
    {
      status = do_cast(from, to_, *X, *Y, shape);
    }
  }

  return status;
}
}  // namespace onnxruntime

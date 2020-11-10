// Copyright 2019 JD.com Inc. JD AI
#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_InternalTesting, _In_ OrtSessionOptions* options);

#ifdef __cplusplus
}
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <set>
#include "core/framework/execution_provider.h"

namespace onnxruntime {
class InternalTestingExecutionProvider : public IExecutionProvider {
 public:
  InternalTestingExecutionProvider(const std::vector<std::string>& ops);
  virtual ~InternalTestingExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  // ??? What do we need to track here ???
  // std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;
  const std::unordered_set<std::string> ops_;
};
}  // namespace onnxruntime

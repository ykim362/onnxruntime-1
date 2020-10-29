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

  common::Status Compile(const std::vector<GraphViewer>& subgraphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  const std::unordered_set<std::string> ops_;
};
}  // namespace onnxruntime

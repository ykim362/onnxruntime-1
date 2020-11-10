// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD_NO_CUSTOM_EPS)

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/framework/fuse_nodes_funcs.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistry;
class KernelRegistryManager;

class GraphPartitioner {
 public:
  enum class Mode {
    kStandard = 0,
    kAssignOnly = 1,    // assign nodes but do not call Compile. used to generate ORT format model with custom EP support
    kOrtFormatLoad = 2  // loading ORT format model. Just partition using custom EPs and GraphViewer based Compile.
  };

  //The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr, const ExecutionProviders& providers)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers) {
    // TODO: What are the steps here:
    // Full build - want to know if we're generating an ORT format model
    // Full and minimal build - when loading from ORT format model, call GetCapability/Compile with GraphViewer
    // instead of creating function node. Want to do in full build as well for testing purposes
    //
    // May want 3 states - normal, creating ORT model, loading ORT model
    // as we will need some mechanism for using a minimal EP that is not device specific but provides only
    // GetCapability on CPU.
  }

  // Run partitioning. Provide compiled_kernel_hashes if mode is kOrtFormatLoad.
  Status Partition(Graph& graph, bool export_dll, FuncManager& func_mgr,
                   Mode mode = Mode::kStandard,
                   std::unordered_map<std::string, uint64_t>* compiled_kernel_hashes = nullptr) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

  Status PartitionOnnxFormatModel(Graph& graph, bool export_dll, FuncManager& func_mgr,
                                  KernelRegistry& fused_kernel_registry, Mode mode) const;

  Status PartitionOrtFormatModel(Graph& graph, FuncManager& func_mgr, KernelRegistry& fused_kernel_registry,
                                 std::unordered_map<std::string, uint64_t>& compiled_kernel_hashes) const;

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
};
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD_NO_CUSTOM_EPS)

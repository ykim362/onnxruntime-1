// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/framework/fuse_nodes_funcs.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelRegistryManager;

class GraphPartitioner {
 public:
  enum class Mode {
    kStandard = 0,
    kAssignOnly = 1,    // assign nodes but do not call Compile. used to generate ORT format model with custom EP support
    kOrtFormatLoad = 2  // loading ORT format model. Just partition using custom EPs and GraphViewer based Compile.
  };

  //The order of providers represents the user preference.
  GraphPartitioner(KernelRegistryManager& kernel_registry_mgr, const ExecutionProviders& providers,
                   Mode mode = Mode::kStandard)
      : kernel_registry_mgr_(kernel_registry_mgr),
        providers_(providers),
        mode_{mode} {
    // TODO: What are the steps here:
    // Full build - want to know if we're generating an ORT format model
    // Full and minimal build - when loading from ORT format model, call GetCapability/Compile with GraphViewer
    // instead of creating function node. Want to do in full build as well for testing purposes
    //
    // May want 3 states - normal, creating ORT model, loading ORT model
    // as we will need some mechanism for using a minimal EP that is not device specific but provides only
    // GetCapability on CPU.
  }

  Status Partition(Graph& graph, bool export_dll, FuncManager& func_mgr) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphPartitioner);

  KernelRegistryManager& kernel_registry_mgr_;
  const ExecutionProviders& providers_;
  Mode mode_;
};
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* INTERNAL_TESTING_EP = "InternalTestingEP";

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops)
    : IExecutionProvider{onnxruntime::kInternalTestingExecutionProvider},
      ops_{ops} {
  // TODO: Allocation planner calls GetAllocator for the individual EP. It would be better if it goes through
  // the session state to get the allocator so it's per-device (or for the allocation planner to try the EP first
  // and fall back to using session state next by passing in a functor it can use to call SessionState::GetAllocator).

  AllocatorCreationInfo device_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(INTERNAL_TESTING_EP,
                                                                    OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

InternalTestingExecutionProvider::~InternalTestingExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  std::unordered_set<std::string> all_node_inputs;
  for (const auto& node : graph_viewer.Nodes()) {
    for (auto* input : node.InputDefs()) {
      all_node_inputs.insert(input->Name());
    }
  }

  /* 
  Very basic search for groups of nodes that can be handled by the EP.
  This doesn't work perfectly if you have a scenario like the following where A and C could be handled by the EP
  but B is between them in the topological sort as you'll get two single node capabilities.

    A  B
    | /   
    C
    |

  */

  const auto& graph_outputs = graph_viewer.GetOutputs();

  std::vector<std::vector<NodeIndex>> node_groups;
  std::vector<NodeIndex> cur_group;
  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    // todo: handle ops with same type in different domains
    if (ops_.count(graph_viewer.GetNode(node_index)->OpType())) {
      cur_group.push_back(node_index);
    } else if (!cur_group.empty()) {
      node_groups.push_back(std::move(cur_group));
    }
  }

  // push any final group
  if (!cur_group.empty()) {
    node_groups.push_back(std::move(cur_group));
  }

  int fused_subgraphs_counter = 0;

  for (const auto& group : node_groups) {
    std::unordered_set<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    // Find inputs and outputs of the subgraph
    std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
    std::unordered_set<const NodeArg*> erased;
    int input_order = 0;
    int output_order = 0;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        const auto it = fused_outputs.find(input);
        if (it != fused_outputs.end()) {
          fused_outputs.erase(it);
          erased.insert(input);
        }
        //only when input is neither in output list nor erased list, add the input to input list
        else if (erased.find(input) == erased.end()) {
          fused_inputs[input] = input_order++;
        }
      }

      // For output searching, there is a special case:
      // If certain output is used more than once,
      // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
      // to the output list

      std::unordered_set<const NodeArg*> processed_outputs;
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto node_idx = it->GetNode().Index();
        const auto* output = node->OutputDefs()[it->GetSrcArgIndex()];

        if (node_set.find(node_idx) != node_set.end()) {
          const auto iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }

        processed_outputs.insert(output);
      }

      for (const auto* output : node->OutputDefs()) {
        if (processed_outputs.find(output) != processed_outputs.end())
          continue;

        const auto iter = fused_inputs.find(output);
        if (iter != fused_inputs.end()) {
          fused_inputs.erase(iter);
          erased.insert(output);
        }
        // only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end() && output->Exists()) {
          fused_outputs[output] = output_order++;
        }
      }
    }

    fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());

    // Sort inputs and outputs by the order they were added
    std::multimap<int, const NodeArg*> inputs, outputs;

    for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
      inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
    }

    for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
      if (all_node_inputs.find(it->first->Name()) != all_node_inputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      } else if (std::find(graph_outputs.begin(), graph_outputs.end(), it->first) != graph_outputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "InternalTestingEP_" + std::to_string(fused_subgraphs_counter++);
    meta_def->domain = kMSDomain;
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

    for (const auto& input : inputs) {
      meta_def->inputs.push_back(input.second->Name());
    }

    for (const auto& output : outputs) {
      meta_def->outputs.push_back(output.second->Name());
    }

    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

common::Status InternalTestingExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                                         std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Create a function to generate dummy empty output for each fused node so the model should be able to be executed.
  for (const auto& node_and_viewer : fused_nodes) {
    NodeComputeInfo compute_info;
    const Node& node = node_and_viewer.fused_node;
    const GraphViewer& graph_viewer = node_and_viewer.filtered_graph;

    // TEMP debug output
    {
      std::cout << "Fusing nodes: ";
      for (const auto& unfused_node : graph_viewer.Nodes()) {
        std::cout << " '" << unfused_node.Name() << "':" << unfused_node.Index();
      }
      std::cout << std::endl;
    }

    compute_info.create_state_func = [&](ComputeContext* /*context*/, FunctionState* /*state*/) {
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
    };

    compute_info.compute_func = [&node](FunctionState /*state*/, const OrtCustomOpApi* c_api,
                                        OrtKernelContext* context) -> Status {
      Ort::CustomOpApi api{*c_api};  // use C++ API for convenience

      const auto outputs = node.OutputDefs();
      const size_t num_outputs = outputs.size();

      for (size_t i = 0; i < num_outputs; i++) {
        const auto* shape_proto = outputs[i]->Shape();
        if (shape_proto == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown output shapes are not supported");
        }

        TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
        if (shape.Size() < 0) {
          // arbitrarily set any unknown dim to 1
          for (size_t idx = 0, end = shape.NumDimensions(); idx < end; ++idx) {
            if (shape[idx] == -1) {
              shape[idx] = 1;
            }
          }
        }

        // create the output_tensor.
        auto* ortvalue = api.KernelContext_GetOutput(context, i, shape.GetDims().data(), shape.GetDims().size());

        // and fill with zeros
        auto* tensor = ortvalue->GetMutable<Tensor>();
        void* data = tensor->MutableDataRaw();
        auto bytes = tensor->SizeInBytes();
        memset(data, 0, bytes);
      };

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/func_kernel.h"

// uncomment this line to count non-CUDA ops in ONNX domain
//#define COUNT_NON_CUDA_OPS

#ifdef COUNT_NON_CUDA_OPS
class NonCudaOps {
 public:
  ~NonCudaOps() {
    printf("Non-CUDA ops:\n");
    for (auto i : map_) {
      printf("%s: %d\n", i.first.c_str(), i.second);
    }
  }

  void AddOp(const std::string& name) {
    if (map_.count(name))
      map_.at(name)++;
    else
      map_.insert({name, 1});
  }

 private:
  std::map<std::string, int> map_;
};

NonCudaOps non_cuda;
#endif

using namespace ::onnxruntime::common;
namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
KernelDefBuilder& BuildFusedKernelDef(KernelDefBuilder& builder, const onnxruntime::Node& node) {
  auto schema = node.Op();
  builder.SetName(schema->Name())
      .SetDomain(schema->domain())
      .SinceVersion(schema->SinceVersion())
      .Provider(node.GetExecutionProviderType());
  return builder;
}

/**
 * Check if a node can be placed on a specific provider.
 * Do nothing if the node is already assigned
 * \param graph
 * \param capability
 * \param kernel_registry_mgr
 * \param provider_type name of the provider to test
 * \param count A counter for generating fused node names. Should be unique within this subgraph
 * \return Fused node. Return nullptr if there is no fuse
 */
static Node* PlaceNode(Graph& graph, std::unique_ptr<IndexedSubGraph> capability,
                       const KernelRegistryManager& kernel_registry_mgr, const std::string& provider_type, int& count,
                       GraphPartitioner::Mode mode) {
  if (nullptr == capability) {
    return nullptr;
  }

  Node* result = nullptr;

  if (nullptr == capability->GetMetaDef()) {
    // The <provider> can run a single node in the <graph> if not using meta-defs.
    // A fused kernel is not supported in this case.
    ORT_ENFORCE(1 == capability->nodes.size());

    auto* node = graph.GetNode(capability->nodes[0]);
    if (nullptr != node && node->GetExecutionProviderType().empty()) {
      // The node was not fused or assigned. Assign it to this <provider>.
      node->SetExecutionProviderType(provider_type);
    }
  } else {
    // The <provider> can run a fused <sub_graph> in the <graph>.

    // Check whether any node in the <sub_graph> was already assigned. If so it cannot be stolen as assignment is done
    // in order of EP priority
    bool sub_graph_available_for_assignment = true;
    for (auto node_index : capability->nodes) {
      auto node = graph.GetNode(node_index);
      if (nullptr == node || !node->GetExecutionProviderType().empty()) {
        // The node was fused or assigned, so that the whole sub-graph will not be assigned to this <provider>
        // The assumption is that this <provider> can only run the sub-graph as a whole unit.
        sub_graph_available_for_assignment = false;
        break;
      }
    }

    if (sub_graph_available_for_assignment) {
      if (mode == GraphPartitioner::Mode::kStandard) {
        std::ostringstream oss;
        oss << provider_type << "_" << capability->GetMetaDef()->name << "_" << count++;
        std::string node_name = oss.str();
        auto& fused_node = graph.FuseSubGraph(std::move(capability), node_name);
        fused_node.SetExecutionProviderType(provider_type);
        // searching in kernel registries, if no kernel registered for the fused_node, use compile approach
        if (!KernelRegistryManager::HasImplementationOf(kernel_registry_mgr, fused_node, provider_type)) {
          result = &fused_node;
        }
      } else {
        // assign the nodes in the indexed subgraph to the current EP so that level 2+ optimizers will not change them.
        // This is used when exporting an ORT format model to maintain the original nodes and re-do the fusion
        // at runtime. The original nodes provide a fallback if fewer nodes can be fused at runtime due to device
        // capabilities.
        for (auto node_index : capability->nodes) {
          auto node = graph.GetNode(node_index);
          // due to above check for sub_graph_available_for_assignment we know 'node' is valid and unassigned
          node->SetExecutionProviderType(provider_type);
        }
      }
    }
  }

  return result;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

Status GraphPartitioner::Partition(Graph& graph, bool export_dll, FuncManager& func_mgr) const {
  // It is a greedy partitioning algorithm per provider preferences user provided when calling ONNX RUNTIME right now.
  // 1. Execution providers' capabilities are checked one by one.
  // 2. All sub-graphs that an execution provider returns will be assigned to it if it's not assigned yet.
  //    NOTE: A 'sub-graph' is a subset of nodes within the current Graph instance.
  //          The control flow nodes have nested Graph instance/s which are also called subgraphs,
  //          but are completely separate Graph instances and not a subset of nodes within a single Graph instance.
  // 3. CPU execution provider is expected to be able to run any node and is the last one in execution provider
  //    preference.
  if (providers_.Empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }

  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  // recurse into nested graphs first so we partition bottom up.
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      // we pass through the export_dll value and FuncManager from the top level graph
      ORT_RETURN_IF_ERROR(Partition(*subgraph, export_dll, func_mgr));
    }
  }

  GraphViewer graph_viewer(graph);

#if !defined(ORT_MINIMAL_BUILD)
  if (mode_ == Mode::kStandard || mode_ == Mode::kAssignOnly) {
    // fused_kernel_registry is preparing the kernels created on the fly for fused sub graph.
    // It is only visible for current session.
    std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();

    // Partitioning <graph> based on provider preference and their capabilities.
    //
    // If an execution provider return the capability that he could run a sub-graph,
    // onnxruntime will fuse the sub-graph into a function node. if the execution provider
    // says it needs to compile the graph at runtime (by returning a MetaDef in ComputeCapability),
    // onnxruntime will invoke the "Compile" method to get compiled binary.
    // There are two mode of compile, one is return the entry point to the compiled binary
    // directly, another is export the compiled binary to shared library for future reuse.

    // TODO: when the graph contain a function node, and user pass in the dll which could
    // run the function by SessionOption, we should create a function kernel for it and
    // delegate the compute to the functions inside the dlls.
    for (auto& provider : providers_) {
      const std::string& type = provider->Type();
      int count = 0;
      std::vector<Node*> nodes_to_compile;
      std::vector<std::unique_ptr<ComputeCapability>> capabilities =
          provider->GetCapability(graph_viewer, kernel_registry_mgr_.GetKernelRegistriesByProviderType(type));

      for (auto& capability : capabilities) {
        Node* n = PlaceNode(graph, std::move(capability->sub_graph), kernel_registry_mgr_, type, count, mode_);
        if (n != nullptr) {
          nodes_to_compile.push_back(n);
        }
      }

      // if mode_ is kAssignOnly, nodes_to_compile will be empty
      if (!nodes_to_compile.empty()) {
        if (export_dll) {
          std::string dll_path;
          ORT_RETURN_IF_ERROR(provider->Compile(nodes_to_compile, dll_path));
          for (auto* node : nodes_to_compile)
            ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), dll_path));
        } else {
          std::vector<NodeComputeInfo> node_compute_funcs;
          ORT_RETURN_IF_ERROR(provider->Compile(nodes_to_compile, node_compute_funcs));
          if (node_compute_funcs.size() != nodes_to_compile.size()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
          }

          for (size_t j = 0; j < nodes_to_compile.size(); j++) {
            ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(nodes_to_compile[j]->Name(), std::move(node_compute_funcs[j])));
          }
        }

        for (auto* node : nodes_to_compile) {
          //prepare the func kernel
          KernelDefBuilder builder;
          BuildFusedKernelDef(builder, *node);
          ORT_RETURN_IF_ERROR(fused_kernel_registry->Register(builder,
                                                              static_cast<KernelCreatePtrFn>(
                                                                  [](const OpKernelInfo& info) -> OpKernel* {
                                                                    return new FunctionKernel(info);
                                                                  })));
        }
      }
    }

    ORT_RETURN_IF_ERROR(graph.Resolve());

    // To see if the node with no provider can be inlined. If one such nodes can be
    // successfully inlined, we re-run the partitioner on the modified graph.
    std::vector<Node*> nodes_need_inline;
    for (auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType().empty()) {
        auto node_func = node.GetFunctionBody();
        if (nullptr == node_func) {
          continue;
        }
        nodes_need_inline.push_back(&node);
      }
    }

    for (auto* node : nodes_need_inline) {
      // If the node has a function body with no kernel and cannot be inlined
      // it is an invalid function
      ORT_RETURN_IF_ERROR(graph.InlineFunction(*node));
    }

    // Resolve and rerun graph partition
    if (!nodes_need_inline.empty()) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
      ORT_RETURN_IF_ERROR(Partition(graph, export_dll, func_mgr));
    }

    //For some cases, like fp16 on cpu, right now we don't have any kernel support that.
    //But we will insert cast op to run the model, so skip the error checking here.
    //If after graph transform phase, the node still not assigned, we will report error
    //during kernel creation phase.
#ifdef COUNT_NON_CUDA_OPS
    for (auto& node : graph.Nodes()) {
      if (node.GetExecutionProviderType() != kCudaExecutionProvider &&
          node.Domain() != kMLDomain &&
          node.Domain() != kMSDomain)
        non_cuda.AddOp(node.OpType());
    }
#endif

    if (!fused_kernel_registry->IsEmpty()) {
      kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry);
    }
  } else
#endif  // !defined(ORT_MINIMAL_BUILD)
  {
    assert(mode_ == Mode::kOrtFormatLoad);

    for (auto& provider : providers_) {
      const std::string& type = provider->Type();
      if (type == kCpuExecutionProvider) {
        // hash for kernel is stored in session state for EPs that have pre-registered kernels
        // (vs. runtime fused kernels) so nothing to do here.
        continue;
      }

      std::vector<std::unique_ptr<ComputeCapability>> capabilities =
          provider->GetCapability(graph_viewer, kernel_registry_mgr_.GetKernelRegistriesByProviderType(type));

      std::vector<IExecutionProvider::FusedNodeAndGraph> fused_nodes;

      int count = 0;
      for (auto& capability : capabilities) {
        const IndexedSubGraph& indexed_sub_graph = *capability->sub_graph;
        const IndexedSubGraph::MetaDef* metadef = indexed_sub_graph.GetMetaDef();
        if (!metadef) {
          // this could happen if we enable another non-ORT EP that has statically assigned kernels.
          // we could add this EP to the skip list above to avoid the unnecessary call to GetCapability as we would
          // have already saved the kernel hashes in the serialized session state.
          continue;
        }

        // TODO: Create a variant of FuseSubGraph where the don't pass in the ownership of ComputeCapability
        // TODO: Create struct with Node and GraphViewer to call Compile with
        //

        std::ostringstream oss;
        oss << type << "_" << metadef->name << "_" << count++;
        std::string node_name = oss.str();
        Node& fused_node = graph.FuseSubGraph(indexed_sub_graph, node_name);
        fused_node.SetExecutionProviderType(type);

        // create filtered graph viewer for this set of nodes
        // TODO: Could avoid the topological sort in the GraphViewer ctor by constructing from an existing
        // GraphViewer instance instead of the Graph (copying the topological order instead of recalculating).
        // Mobile models are expected to be small though so shouldn't be a significant perf cost for now.
        fused_nodes.emplace_back(
            IExecutionProvider::FusedNodeAndGraph{&fused_node, GraphViewer(graph, indexed_sub_graph)});
      }

      std::vector<NodeComputeInfo> node_compute_funcs;
      ORT_RETURN_IF_ERROR(provider->Compile(fused_nodes, node_compute_funcs));

      if (node_compute_funcs.size() != fused_nodes.size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
      }

      for (size_t j = 0; j < fused_nodes.size(); j++) {
        ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(fused_nodes[j].fused_node->Name(), std::move(node_compute_funcs[j])));
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

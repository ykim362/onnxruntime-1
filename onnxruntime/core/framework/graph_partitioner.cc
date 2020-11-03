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

static KernelDefBuilder& BuildFusedKernelDef(KernelDefBuilder& builder, const IndexedSubGraph::MetaDef& metadef,
                                             const std::string& provider_type) {
  builder.SetName(metadef.name)
      .SetDomain(metadef.domain)
      .SinceVersion(metadef.since_version)
      .Provider(provider_type);
  return builder;
}

#if !defined(ORT_MINIMAL_BUILD)
static KernelDefBuilder& BuildFusedKernelDef(KernelDefBuilder& builder, const onnxruntime::Node& node) {
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
static Node* PlaceNode(Graph& graph, std::unique_ptr<IndexedSubGraph>& capability,
                       const KernelRegistryManager& kernel_registry_mgr, const std::string& provider_type, int& count,
                       IExecutionProvider::FusionStyle fusion_style,
                       GraphPartitioner::Mode mode) {
  if (nullptr == capability) {
    return nullptr;
  }

  Node* result = nullptr;
  bool release_capability = true;

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
        // if mode is kAssignOnly we want all nodes that can _potentially_ be taken by compiling EPs to be assigned,
        // so that we aggregate the nodes covered and ensure the original nodes remain in the ORT format model by
        // preventing level 2 and 3 optimizers from changing them (optimizers check the EP the node is assigned to
        // and only make changes if the EP is on the optimizer's list of supported EPs. an EP that compiles nodes
        // should never be on those lists).
        //
        // when the ORT format model is loaded we will process it normally with EP priority being applied for
        // whichever EPs are enabled at the time.
        //
        //e.g. an Android NNAPI EP may take different/overlapping nodes to a iOS CoreML EP.
        // We want the ORT format model to be able to be run as efficiently as possible on either platform,
        // so we want all the nodes that either may take to be preserved. If we did not do this we would
        // need to create one ORT format model for Android and one for iOS.
        if (mode != GraphPartitioner::Mode::kAssignOnly) {
          // The node was fused or assigned, so that the whole sub-graph will not be assigned to this <provider>
          // The assumption is that this <provider> can only run the sub-graph as a whole unit.
          sub_graph_available_for_assignment = false;
          break;
        }
      }
    }

    if (sub_graph_available_for_assignment) {
      if (mode == GraphPartitioner::Mode::kStandard) {
        std::ostringstream oss;
        oss << provider_type << "_" << capability->GetMetaDef()->name << "_" << count++;
        std::string node_name = oss.str();

        Node* fused_node = nullptr;
        if (fusion_style == IExecutionProvider::FusionStyle::Function) {
          fused_node = &graph.FuseSubGraph(std::move(capability), node_name);
        } else {
          // create a fused node without copying everything to a Function body. The IndexedSubGraph will be passed
          // through to Compile via a filtered GraphViewer so we don't transfer ownership of it here.
          release_capability = false;
          fused_node = &graph.BeginFuseSubGraph(*capability, node_name);
        }

        fused_node->SetExecutionProviderType(provider_type);

        // searching in kernel registries, if no kernel registered for the fused_node, use compile approach
        if (!KernelRegistryManager::HasImplementationOf(kernel_registry_mgr, *fused_node, provider_type)) {
          result = fused_node;
        }
      } else {
        assert(mode == GraphPartitioner::Mode::kAssignOnly);

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

  if (release_capability) {
    capability = nullptr;
  }

  return result;
}

Status GraphPartitioner::InlineNodes(Graph& graph, bool export_dll, FuncManager& func_mgr) const {
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

  return Status::OK();
}

Status GraphPartitioner::PartitionOnnxFormatModel(Graph& graph, bool export_dll, FuncManager& func_mgr,
                                                  Mode mode) const {
  GraphViewer graph_viewer(graph);

  // fused_kernel_registry is preparing the kernels created on the fly for fused sub graph.
  // It is only visible for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();

  // Partitioning <graph> based on provider preference and their capabilities.
  //
  // If an execution provider returns the capability that he could run a sub-graph,
  // onnxruntime will fuse the sub-graph into a function node. if the execution provider
  // says it needs to compile the graph at runtime (by returning a MetaDef in ComputeCapability),
  // onnxruntime will invoke the "Compile" method.
  // There are two mode of compile, one is return the entry point to the compiled binary
  // directly or a local function, another is to export the compiled binary to shared library for future reuse.

  // TODO: when the graph contain a function node, and user pass in the dll which could
  // run the function by SessionOption, we should create a function kernel for it and
  // delegate the compute to the functions inside the dlls.
  for (auto& provider : providers_) {
    const std::string& type = provider->Type();
    auto fusion_style = provider->GetFusionStyle();
    int count = 0;
    std::vector<Node*> nodes_to_compile;

    std::vector<std::unique_ptr<ComputeCapability>> capabilities =
        provider->GetCapability(graph_viewer, kernel_registry_mgr_.GetKernelRegistriesByProviderType(type));

    std::vector<std::unique_ptr<ComputeCapability>> capabilities_to_compile;
    capabilities_to_compile.reserve(std::count_if(capabilities.cbegin(), capabilities.cend(),
                                                  [](const std::unique_ptr<ComputeCapability>& entry) {
                                                    return entry->sub_graph != nullptr &&
                                                           entry->sub_graph->GetMetaDef() != nullptr;
                                                  }));

    for (auto& capability : capabilities) {
      Node* n = PlaceNode(graph, capability->sub_graph, kernel_registry_mgr_, type, count, fusion_style, mode);
      if (n != nullptr) {
        nodes_to_compile.push_back(n);
        capabilities_to_compile.push_back(std::move(capability));
      }
    }

    // NOTE: if mode_ is kAssignOnly, nodes_to_compile will be empty
    if (!nodes_to_compile.empty()) {
      std::vector<NodeComputeInfo> node_compute_funcs;

      if (export_dll) {
        ORT_ENFORCE(fusion_style == IExecutionProvider::FusionStyle::Function,
                    "Must use Function based fusion when exporting compiled nodes to dll.");
      }

      if (fusion_style == IExecutionProvider::FusionStyle::Function) {
        // Create a Function based node where the fused nodes have a new Graph instance.

        if (export_dll) {
          std::string dll_path;
          ORT_RETURN_IF_ERROR(provider->Compile(nodes_to_compile, dll_path));
          for (auto* node : nodes_to_compile) {
            ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), dll_path));
          }
        } else {
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

      } else {
        // storage for the GraphViewer for each IndexedSubGraph
        std::vector<std::unique_ptr<GraphViewer>> viewers;
        viewers.reserve(nodes_to_compile.size());
        std::vector<IExecutionProvider::FusedNodeAndGraph> nodes_and_viewers;

        ORT_ENFORCE(nodes_to_compile.size() == capabilities_to_compile.size(),
                    "Internal error. Mismatch between number of fused nodes and compute capabilities instances.");

        for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
          auto* node = nodes_to_compile[j];
          const auto& cur_capability = capabilities_to_compile[j];
          viewers.push_back(onnxruntime::make_unique<GraphViewer>(graph, *cur_capability->sub_graph));
          nodes_and_viewers.push_back(IExecutionProvider::FusedNodeAndGraph{*node, *viewers.back()});
        }

        ORT_RETURN_IF_ERROR(provider->Compile(nodes_and_viewers, node_compute_funcs));
        if (node_compute_funcs.size() != nodes_to_compile.size()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
        }

        for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
          auto* node = nodes_to_compile[j];
          const auto& cur_capability = capabilities_to_compile[j];

          //
          // ??? we use Node.Name here but metadef.name for the compiled kernel. is that correct???
          //
          ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), std::move(node_compute_funcs[j])));

          // create the func kernel for the name in the MetaDef. this is also the node name and that name that will
          // used as the key in the FuncManager entry. We need the registry to own the KernelCreateInfo that is
          // used by SessionState
          KernelDefBuilder builder;
          BuildFusedKernelDef(builder, *cur_capability->sub_graph->GetMetaDef(), type);
          ORT_RETURN_IF_ERROR(fused_kernel_registry->Register(builder,
                                                              static_cast<KernelCreatePtrFn>(
                                                                  [](const OpKernelInfo& info) -> OpKernel* {
                                                                    return new FunctionKernel(info);
                                                                  })));

          // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
          graph.FinalizeFuseSubGraph(*cur_capability->sub_graph, *node);
        }
      }
    }
  }

  ORT_RETURN_IF_ERROR(graph.Resolve());
  ORT_RETURN_IF_ERROR(InlineNodes(graph, export_dll, func_mgr));

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

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

// Simplified partitioning where custom EPs may produce compiled nodes.
// EPs with static kernels do not need to be processed as their kernels are matched via hash information serialized
// as part of the ORT format model.
Status GraphPartitioner::PartitionOrtFormatModel(
    Graph& graph, bool /*export_dll*/, FuncManager& func_mgr,
    std::unordered_map<std::string, uint64_t>& compiled_kernel_hashes) const {
  GraphViewer graph_viewer(graph);
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();

  for (auto& provider : providers_) {
    const std::string& type = provider->Type();
    if (type == kCpuExecutionProvider) {
      // hash for kernel is stored in session state for EPs that have pre-registered kernels
      // (vs. runtime fused kernels) so nothing to do here.
      continue;
    }

    //
    // TODO: Some duplication with ONNX format model partitioning when mode is FilteredGraphViewer. Refactor to reuse
    // but need to extract out some parts of PlaceNode and some parts of the partitioning.
    //

    std::vector<std::unique_ptr<ComputeCapability>> capabilities =
        provider->GetCapability(graph_viewer, kernel_registry_mgr_.GetKernelRegistriesByProviderType(type));

    std::vector<IExecutionProvider::FusedNodeAndGraph> nodes_and_viewers;

    int count = 0;

    // storage for the GraphViewer for each IndexedSubGraph
    std::vector<std::unique_ptr<GraphViewer>> viewers;
    viewers.reserve(capabilities.size());

    bool skip_ep = false;
    for (auto& capability : capabilities) {
      const IndexedSubGraph& indexed_sub_graph = *capability->sub_graph;
      const IndexedSubGraph::MetaDef* metadef = indexed_sub_graph.GetMetaDef();
      if (!metadef) {
        // this could happen if we enable another non-ORT EP that has statically assigned kernels.
        // we could add this EP to the skip list above to avoid the unnecessary call to GetCapability as we would
        // have already saved the kernel hashes in the serialized session state.
        skip_ep = true;
        break;
      }

      std::ostringstream oss;
      oss << type << "_" << metadef->name << "_" << count++;
      std::string node_name = oss.str();

      Node& fused_node = graph.BeginFuseSubGraph(indexed_sub_graph, node_name);
      fused_node.SetExecutionProviderType(type);

      // create filtered graph viewer for this set of nodes
      //
      // TODO: Could avoid the topological sort in the GraphViewer ctor by constructing from an existing
      // GraphViewer instance instead of the Graph (copying the topological order instead of recalculating).
      viewers.push_back(onnxruntime::make_unique<GraphViewer>(graph, indexed_sub_graph));
      nodes_and_viewers.push_back(IExecutionProvider::FusedNodeAndGraph{fused_node, *viewers.back()});
    }

    if (skip_ep) {
      continue;
    }

    std::vector<NodeComputeInfo> node_compute_funcs;
    ORT_RETURN_IF_ERROR(provider->Compile(nodes_and_viewers, node_compute_funcs));

    if (node_compute_funcs.size() != nodes_and_viewers.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
    }

    for (size_t j = 0; j < nodes_and_viewers.size(); j++) {
      Node& node = nodes_and_viewers[j].fused_node;
      const auto& cur_capability = capabilities[j];
      const IndexedSubGraph& indexed_sub_graph = *cur_capability->sub_graph;
      const IndexedSubGraph::MetaDef& metadef = *indexed_sub_graph.GetMetaDef();

      ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node.Name(),
                                               std::move(node_compute_funcs[j])));

      KernelDefBuilder builder;
      BuildFusedKernelDef(builder, metadef, type);
      auto kernel_def = builder.Build();

      // save hash so SessionState can find the kernel
      compiled_kernel_hashes.insert({metadef.name, kernel_def->GetHash()});

      ORT_RETURN_IF_ERROR(fused_kernel_registry->Register(
          KernelCreateInfo(std::move(kernel_def), static_cast<KernelCreatePtrFn>(
                                                      [](const OpKernelInfo& info) -> OpKernel* {
                                                        return new FunctionKernel(info);
                                                      }))));

      // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
      graph.FinalizeFuseSubGraph(indexed_sub_graph, node);
    }
  }

  if (!fused_kernel_registry->IsEmpty()) {
    kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry);
  }

  return Status::OK();
}

Status GraphPartitioner::Partition(Graph& graph, bool export_dll, FuncManager& func_mgr, Mode mode,
                                   std::unordered_map<std::string, uint64_t>* compiled_kernel_hashes) const {
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

  if (mode == Mode::kStandard || mode == Mode::kAssignOnly) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_RETURN_IF_ERROR(PartitionOnnxFormatModel(graph, export_dll, func_mgr, mode));
#else
    ORT_THROW("Not supported in this build.");
#endif
  } else {
    ORT_ENFORCE(compiled_kernel_hashes != nullptr, "Compiled kernel hashes were not provided");

    ORT_RETURN_IF_ERROR(PartitionOrtFormatModel(graph, export_dll, func_mgr, *compiled_kernel_hashes));
  }

  return Status::OK();
}
}  // namespace onnxruntime

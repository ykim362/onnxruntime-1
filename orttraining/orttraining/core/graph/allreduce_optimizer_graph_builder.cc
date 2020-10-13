// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"

namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

static Status AddNcclAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    ArgDef& fused_gradient_argdef,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& fused_allreduce_output) {
  fused_allreduce_output = ArgDef(fused_gradient_argdef.name + "AllReduce_Out", fused_gradient_argdef.type_proto);

  // Add NCCL Allreduce node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  {fused_gradient_argdef},
                                  {fused_allreduce_output},
                                  NodeAttributes(),
                                  "NcclAllReduce")});

  std::vector<ArgDef> view_inputs(gradient_argdefs.size() + 1);
  view_inputs[0] = fused_allreduce_output;

  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    ArgDef& gradient_shape = view_inputs[i + 1];
    gradient_shape = ArgDef(gradient_argdefs[i].name + "_Shape");

    graph_defs.AddNodeDefs({NodeDef("Shape",
                                    {gradient_argdefs[i]},
                                    {gradient_shape},
                                    NodeAttributes(),
                                    gradient_shape.name)});
  }

  std::vector<ArgDef> allreduce_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    TypeProto* allreduced_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[i]);
    allreduced_gradient_type_proto->mutable_tensor_type()->set_elem_type(
        fused_gradient_argdef.type_proto->tensor_type().elem_type());

    allreduce_outputs[i] = ArgDef(gradient_argdefs[i].name + "_AllReduce_Out", allreduced_gradient_type_proto);
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"View", kMSDomain, 1},
                                  view_inputs,
                                  allreduce_outputs,
                                  NodeAttributes(),
                                  "AllReduceOutputView")});

  gradient_argdefs = allreduce_outputs;
  return Status::OK();
}

static std::vector<ArgDef> GetGradientNormInputs(
    const std::vector<ArgDef>& gradient_argdefs,
    ArgDef fused_gradient_argdef) {
  if (!fused_gradient_argdef.name.empty()) {
    return {fused_gradient_argdef};
  } else {
    return gradient_argdefs;
  }
}

AllreduceOptimizerGraphBuilder::AllreduceOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1,
              "Allreduce optimizer graph builder can only be used for distributed training.");
  if (opt_graph_config.use_nccl) {
    ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
  } else {
    ORT_THROW("Performing Allreduce is only supported using NCCL.");
  }
}

Status AllreduceOptimizerGraphBuilder::BuildInternal(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  // add gradient scaling
  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations =
      opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  const float scale = 1.0f / total_num_accumulations;
  const bool fuse_scaling_outputs = opt_graph_config_.use_nccl;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), fuse_scaling_outputs));

  // add Allreduce for gradients
  ArgDef reduced_fused_gradient_argdef;

  ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradients(gradient_argdefs, fused_gradient_argdef, graph_defs, reduced_fused_gradient_argdef));

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    auto gradient_norm_inputs = GetGradientNormInputs(gradient_argdefs, reduced_fused_gradient_argdef);
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;

    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, {global_grad_norm_argdef}, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GradientAllIsFinite] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
      opt_builder_registry_, weight_argdefs, gradient_argdefs,
      &global_grad_norm_argdef,
      &global_grad_norm_finite_argdef,
      opt_configs_, graph_defs,
      optimizer_state_initializer_names));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime

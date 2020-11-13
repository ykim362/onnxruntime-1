// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"

#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

namespace onnxruntime{
namespace test {
namespace {
  constexpr auto ORIGINAL_MODEL_PATH = ORT_TSTR("testdata/test_training_model.onnx");
  constexpr auto BACKWARD_MODEL_PATH = ORT_TSTR("testdata/temp_backward_model.onnx");
  
  std::unordered_set<std::string> GetModelOutputNames(const InferenceSession& session) {
  const auto outputs_result = session.GetModelOutputs();
  ORT_ENFORCE(outputs_result.first.IsOK(), "Failed to get model outputs: ", outputs_result.first.ErrorMessage());
  std::unordered_set<std::string> output_names{};
  for (const auto* output : *outputs_result.second) {
    output_names.insert(output->Name());
  }
  return output_names;
}
} // namespace

static TrainingSession::TrainingConfiguration MakeBasicTrainingConfig() {
  TrainingSession::TrainingConfiguration config{};
  config.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  config.loss_function_config = TrainingSession::TrainingConfiguration::LossFunctionConfiguration{};
  config.loss_function_config.value().loss_function_info =
      LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});
  //config.weight_names_to_train = {"B3", "W2"};

  return config;
}

static Status BuildBackPropGraph(
    const PathString& forward_model_file,
    const TrainingSession::TrainingConfiguration& config,
    PathString& backward_model_file,
    TrainingSession& training_session) {
  // std::unique_ptr<Environment> env;
  // ORT_RETURN_IF_ERROR(Environment::Create(nullptr, env));

   SessionOptions so{};
  // training_session = TrainingSession{so, *env};

  std::cout << "Loading source model file = " << ToMBString(forward_model_file) << "\n";

  ORT_RETURN_IF_ERROR(training_session.Load(forward_model_file));

  TrainingSession::TrainingConfigurationResult config_result{};
  ORT_RETURN_IF_ERROR(training_session.ConfigureForTraining(config, config_result));

  backward_model_file = config.model_with_training_graph_path.value();

  ORT_RETURN_IF_ERROR(training_session.Initialize());

  std::cout << "After training_session->Initialize" << std::endl;
  ORT_THROW_IF_ERROR(training_session.PrintWeights());

  std::vector<MLValue> gradient_fetches;
  RunOptions run_options;
  //run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_log_verbosity_level = 0;
  run_options.run_tag = so.session_logid;
  run_options.training_mode = true;

  // Create dummy feeds
  std::vector<int64_t> image_dims = {1, 784};
  std::vector<int64_t> label_dims = {1, 10};
  std::vector<float> image_value(784, 1);
  std::vector<float> label_value(10, 1);

  MLValue imageMLValue;
  TrainingUtil::CreateCpuMLValue(image_dims, image_value, &imageMLValue);
  MLValue labelMLValue;
  TrainingUtil::CreateCpuMLValue(label_dims, label_value, &labelMLValue);

  auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X", "labels"}, {imageMLValue, labelMLValue});
  
  auto output_names_include_gradients = GetModelOutputNames(training_session);
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  auto start_time = std::chrono::high_resolution_clock::now();

  std::cout << "Before training_session->Run" << std::endl;
  ORT_THROW_IF_ERROR(training_session.PrintWeights());

  ORT_THROW_IF_ERROR(training_session.Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches));

  std::cout << "After training_session.Run" << std::endl;
  ORT_THROW_IF_ERROR(training_session.PrintWeights());

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed = TimeDiffMicroSeconds(start_time, end_time);
  std::cout << "Training session run completed in " << elapsed << " microseconds.\n";

  return Status::OK();
}

/**
 * Run a training session for this model for 1 epoch, using batch size of 1 and synthetic input data.
 * @param so - SessionOptions for this run.
 * @param backprop_model_file - Model file to be run. This should already contain loss function and backward prop nodes.
 * @return TrainingSession for this run.
 */
static std::unique_ptr<TrainingSession> RunTrainingSessionWithChecks(
  const SessionOptions& so, const PathString& backprop_model_file) {
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  std::unique_ptr<TrainingSession> training_session = onnxruntime::make_unique<TrainingSession>(so, *env);

  ORT_THROW_IF_ERROR(training_session->Load(backprop_model_file));

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  ORT_THROW_IF_ERROR(res.first);
  ORT_ENFORCE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << '\n';

  std::cout << "Before training_session->Initialize" << std::endl;
  ORT_THROW_IF_ERROR(training_session->PrintWeights());

  ORT_THROW_IF_ERROR(training_session->Initialize());

  std::cout << "After training_session->Initialize" << std::endl;
  ORT_THROW_IF_ERROR(training_session->PrintWeights());

  std::vector<MLValue> gradient_fetches;
  RunOptions run_options;
  //run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_log_verbosity_level = 0;
  run_options.run_tag = so.session_logid;
  run_options.training_mode = true;

  // Create dummy feeds
  std::vector<int64_t> image_dims = {1, 784};
  std::vector<int64_t> label_dims = {1, 10};
  std::vector<float> image_value(784, 1);
  std::vector<float> label_value(10, 1);

  MLValue imageMLValue;
  TrainingUtil::CreateCpuMLValue(image_dims, image_value, &imageMLValue);
  MLValue labelMLValue;
  TrainingUtil::CreateCpuMLValue(label_dims, label_value, &labelMLValue);

  //auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X", "predictions"}, {imageMLValue, labelMLValue});
  auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X"}, {imageMLValue});

  auto output_names_include_gradients = GetModelOutputNames(*training_session);
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  auto start_time = std::chrono::high_resolution_clock::now();

  std::cout << "Before training_session->Run" << std::endl;
  ORT_THROW_IF_ERROR(training_session->PrintWeights());

  ORT_THROW_IF_ERROR(training_session->Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches));

  std::cout << "After training_session Run" << std::endl;
  ORT_THROW_IF_ERROR(training_session->PrintWeights());

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed = TimeDiffMicroSeconds(start_time, end_time);
  std::cout << "Training session run completed in " << elapsed << " microseconds.\n";

  return training_session;
}

TEST(TrainingSessionTest, BasicTraining) {
  const auto config = MakeBasicTrainingConfig();
  PathString backprop_model_file;
  std::unique_ptr<Environment> env;
  Environment::Create(nullptr, env);

  SessionOptions so{};
  TrainingSession training_session{so, *env};
  ASSERT_STATUS_OK(BuildBackPropGraph(ORIGINAL_MODEL_PATH, config, backprop_model_file, training_session));

  NameMLValMap state_tensors;
  ORT_THROW_IF_ERROR(training_session.GetStateTensors(state_tensors));
  std::cout << state_tensors.size() << std::endl;
  for (auto& kv : state_tensors) {
    auto& rtensor = kv.second.Get<Tensor>();
    std::cout << kv.first << " " << &rtensor.Shape() << std::endl;
  }
}
}
}
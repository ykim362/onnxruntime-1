#include "core/common/logging/logging.h"
#include "core/providers/internal_testing/internal_testing_execution_provider.h"
#include "core/session/inference_session.h"
// #include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

TEST(InternalTestingEP, TestBasicFunctionality) {
  // arbitrary test model
  std::string model_path("testdata/test_training_model.onnx");

  SessionOptions so;
  InferenceSessionWrapper session(so, GetEnvironment());

  // set supported ops to MatMul and Add as these are found in pairs through the model.
  // we can say the EP potentially handles them all, but can also simulate removing handling of Add
  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
  // that we can revert ops back to the CPU implementation
  //
  const std::vector<std::string> supported_ops{"MatMul", "Add"};

  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultInternalTestingExecutionProvider(supported_ops)));
  ASSERT_STATUS_OK(session.Load(model_path));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  for (const auto& node : graph.Nodes()) {
    std::cout << "Node:'" << node.Name() << "' OpType:" << node.OpType()
              << " EP:" << node.GetExecutionProviderType() << "\n";
  }

  ASSERT_EQ(graph.NumberOfNodes(), 5);

  OrtValue ml_value_x;
  TensorShape input_shape{1, 784};
  std::vector<float> input(input_shape.Size(), 0.f);

  CreateMLValue<float>(input_shape.GetDims(), input.data(), OrtMemoryInfo(), &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("predictions");
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
}

TEST(InternalTestingEP, TestSaveAndLoadOrtModel) {}
TEST(InternalTestingEP, TestLoadOrtModel) {}
TEST(InternalTestingEP, TestMinimalRegistrationOfEPwithGetCapability) {}
TEST(InternalTestingEP, TestModelWithSubgraph) {
}

}  // namespace test
}  // namespace onnxruntime

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
  std::string model_path("testdata/mnist.onnx");

  SessionOptions so;
  InferenceSessionWrapper session(so, GetEnvironment());

  // set supported ops to some ops that are found consecutively in the model.
  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
  // that we can revert ops back to the CPU implementation successfully
  //
  const std::vector<std::string> supported_ops{"Conv", "Add", "Relu", "MaxPool"};

  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultInternalTestingExecutionProvider(supported_ops)));
  ASSERT_STATUS_OK(session.Load(model_path));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  for (const auto& node : graph.Nodes()) {
    std::cout << "Node:'" << node.Name() << "' OpType:" << node.OpType()
              << " EP:" << node.GetExecutionProviderType() << "\n";
  }

  // the custom EP covers the 2 Conv -> Add-> Relu -> Maxpool sequences in a single op.
  // the reshape of an initializer gets constant folded, and MatMul + Add is fused into FusedGemm
  // leaving 3 nodes once we're done.
  ASSERT_EQ(graph.NumberOfNodes(), 3);

  OrtValue ml_value_x;
  TensorShape input_shape{1, 1, 28, 28};
  std::vector<float> input(input_shape.Size(), 1.f);

  CreateMLValue<float>(input_shape.GetDims(), input.data(), OrtMemoryInfo(), &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Plus214_Output_0");
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

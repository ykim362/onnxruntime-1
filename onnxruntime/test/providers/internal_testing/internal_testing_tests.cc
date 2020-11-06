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
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

static void CreateSession(const SessionOptions& so, std::unique_ptr<InferenceSessionWrapper>& session,
                          std::string model_path = "testdata/mnist.onnx",  // arbitrary test model
                          const std::vector<std::string>* override_supported_ops = nullptr) {
  session = onnxruntime::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  // set supported ops to some ops that are found consecutively in the model.
  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
  // that we can revert ops back to the CPU implementation successfully
  const std::vector<std::string> default_supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  const std::vector<std::string>* supported_ops = override_supported_ops ? override_supported_ops
                                                                         : &default_supported_ops;

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(DefaultInternalTestingExecutionProvider(*supported_ops)));
  ASSERT_STATUS_OK(session->Load(model_path));
  ASSERT_STATUS_OK(session->Initialize());
}
TEST(InternalTestingEP, TestBasicFunctionality) {
  std::unique_ptr<InferenceSessionWrapper> session;
  CreateSession(SessionOptions{}, session);

  const auto& graph = session->GetGraph();

  //for (const auto& node : graph.Nodes()) {
  //  std::cout << "Node:'" << node.Name() << "' OpType:" << node.OpType()
  //            << " EP:" << node.GetExecutionProviderType() << "\n";
  //}

  // the custom EP covers the 2 Conv -> Add-> Relu -> Maxpool sequences in a single op.
  // the reshape of an initializer gets constant folded, and MatMul + Add is fused into FusedGemm
  // leaving 3 nodes once we're done.
  ASSERT_EQ(graph.NumberOfNodes(), 3);

  // validate that we can execute the model. the dummy internal testing EP just creates empty output so the
  // values in the output aren't relevant. all we care about is that we can execute the model and produce output.
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

  ASSERT_STATUS_OK(session->Run(feeds, output_names, &fetches));

  // check that the output is all zeros. the dummy EP produces output of the correct shape will all zeros, so any
  // downstream operations should still result in zeros for this model
  // OR it should equal the bias in the final Add operation, which is in the Parameter194 initializer
  const auto& t = fetches[0].Get<Tensor>();
  const auto data = t.DataAsSpan<float>();

  const auto& session_state = session->GetSessionState();
  int idx = 0;
  ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx("Parameter194", idx));
  const auto& initializer = session_state.GetConstantInitializedTensors().at(idx);
  const auto expected = initializer.Get<Tensor>().DataAsSpan<float>();

  ASSERT_THAT(data, ::testing::ContainerEq(expected));
}

TEST(InternalTestingEP, TestSaveAndLoadOrtModel) {
  std::unique_ptr<InferenceSessionWrapper> session;
  SessionOptions so;
  so.optimized_model_filepath = ORT_TSTR("testdata/mnist.ort");

  CreateSession(so, session);

  // we should now have an mnist.ort file
  std::unique_ptr<InferenceSessionWrapper> session2;
  CreateSession(SessionOptions{}, session2, "testdata/mnist.ort");
  const auto& graph = session2->GetGraph();
  ASSERT_EQ(graph.NumberOfNodes(), 3);
}

// test to validate a minimal build
TEST(InternalTestingEP, TestLoadOrtModel) {}

// test that is the custom EP cannot take all nodes due to device limitations
// that we fallback to the CPU implementations and can execute the model
TEST(InternalTestingEP, TestLoadOrtModelWithReducedOpCoverage) {}

TEST(InternalTestingEP, TestMinimalRegistrationOfEPwithGetCapability) {}
TEST(InternalTestingEP, TestModelWithSubgraph) {
}

}  // namespace test
}  // namespace onnxruntime

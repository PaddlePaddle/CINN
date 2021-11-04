#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "infrt/api/infrt_api.h"
#include "infrt/common/buffer.h"
#include "infrt/common/dtype.h"
#include "llvm/Support/raw_ostream.h"

using infrt::CinnRtConfig;
using infrt::CinnRtPredictor;
using infrt::CreateCinnRtPredictor;

namespace infrt {

TEST(CinnRtPredictor, predictor) {
  std::vector<std::string> shared_libs;
  shared_libs.push_back("../../paddle/libexternal_kernels.so");

  CinnRtConfig config;

  // set external shared libraries that contain kernels.
  config.set_shared_libs(shared_libs);
  // set model dir
  config.set_model_dir("../../paddle/paddle_1.8_fc_model");
  // set mlir path
  config.set_mlir_path("../../../infrt/dialect/mlir_tests/tensor_map.mlir");

  std::shared_ptr<CinnRtPredictor> predictor = CreateCinnRtPredictor(config);

  auto* input                = predictor->GetInput(0);
  std::vector<int64_t> shape = {3, 3};
  input->Init(shape, infrt::GetDType<float>());
  llvm::outs() << input->shape() << "\n";

  // init input tensor
  auto* input_data = reinterpret_cast<float*>(input->buffer()->data()->memory);
  for (int i = 0; i < input->shape().GetNumElements(); i++) input_data[i] = 1.0;

  predictor->Run();

  // get and print output tensor
  auto* output      = predictor->GetOutput(0);
  auto* output_data = reinterpret_cast<float*>(output->buffer()->data()->memory);

  std::vector<float> ans = {0.428458, 0.244493, 0.572342, 0.572008, 0.509771, 0.495599, 0.651287, 0.326426, 0.404649};

  ASSERT_EQ(output->shape().GetNumElements(), ans.size());
  for (int i = 0; i < output->shape().GetNumElements(); ++i) {
    ASSERT_NEAR(output_data[i], ans[i], 0.000001);
  }
}

}  // namespace infrt

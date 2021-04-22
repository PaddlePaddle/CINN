#include <iostream>

#include "cinn/hlir/framework/buffer.h"
#include "cinnrt/api/cinnrt_api.h"
#include "cinnrt/common/dtype.h"
#include "llvm/Support/raw_ostream.h"

using cinnrt::CinnRtConfig;
using cinnrt::CinnRtPredictor;
using cinnrt::CreateCinnRtPredictor;

int main() {
  std::vector<std::string> shared_libs;
  shared_libs.push_back("/cinn/paddle/libexternal_kernels.so");

  CinnRtConfig config;
  // set external shared libraries that contain kernels.
  config.set_shared_libs(shared_libs);
  // set model dir
  config.set_model_dir("/cinn/benchmark/paddle-inference/Paddle-Inference-Demo/c++/fc/fc_1.8");
  // set mlir path
  config.set_mlir_path("/cinn/cinnrt/dialect/mlir_tests/tensor_map_predictor.mlir");

  std::shared_ptr<CinnRtPredictor> predictor = CreateCinnRtPredictor(config);

  // std::cout << "input num: " << predictor->GetInputNum() << std::endl;
  // std::cout << "output num: " << predictor->GetOutputNum() << std::endl;
  auto* input                = predictor->GetInput(0);
  std::vector<int64_t> shape = {3, 3};
  input->Init(shape, cinnrt::GetDType<float>());
  llvm::outs() << input->shape() << "\n";

  // init input tensor
  auto* input_data = reinterpret_cast<float*>(input->buffer()->data()->memory);
  for (int i = 0; i < input->shape().GetNumElements(); i++) input_data[i] = 1.0;

  predictor->Run();

  // get and print output tensor
  auto* output      = predictor->GetOutput(0);
  auto* output_data = reinterpret_cast<float*>(output->buffer()->data()->memory);
  std::cout << "output tensor: [";
  for (int i = 0; i < output->shape().GetNumElements(); ++i) std::cout << output_data[i] << ", ";
  std::cout << "]" << std::endl;

  return 0;
}

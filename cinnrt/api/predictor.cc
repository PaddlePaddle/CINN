#include "cinnrt/api/cinnrt_api.h"

using cinnrt::CinnrtConfig;
using cinnrt::CinnrtPredictor;
using cinnrt::CreateCinnrtPredictor;

int main() {
  std::vector<std::string> shared_libs;
  shared_libs.push_back("/cinn/paddle/libexternal_kernels.so");

  CinnrtConfig config;
  config.set_shared_libs(shared_libs);
  config.set_model_dir("/cinn/benchmark/paddle-inference/Paddle-Inference-Demo/c++/fc/fc_1.8");
  config.set_mlir_path("/cinn/cinnrt/dialect/mlir_tests/tensor_map_predictor.mlir");

  std::shared_ptr<CinnrtPredictor> predictor = CreateCinnrtPredictor(config);
  predictor->Run();
  return 0;
}

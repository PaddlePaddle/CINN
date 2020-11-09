#pragma once

#include <string>
#include <vector>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace tests {

class OpBenchmarkTester {
 public:
  OpBenchmarkTester(const std::string &op_name,
                    const std::vector<std::vector<int>> &input_shapes,
                    const common::Target &target = common::DefaultHostTarget(),
                    int repeat                   = 10,
                    float diff                   = 1e-5)
      : op_name_(op_name), input_shapes_(input_shapes), target_(target), repeat_(repeat), diff_(diff) {}

  virtual ~OpBenchmarkTester() = default;

  void CreateBuffer();

  void TestOp(const std::string &test_name,
              const hlir::framework::NodeAttr &attrs,
              const std::vector<Type> &out_types,
              bool use_default_stragegy = true);

  Module CreateCinnModule(const hlir::framework::NodeAttr &attrs,
                          const std::vector<Type> &out_types,
                          bool use_default_stragegy = true);

  // should define specific stragey if not use default schedule
  virtual std::vector<ir::Tensor> CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                         poly::StageMap *stages) {
    CINN_NOT_IMPLEMENTED
  }

  auto CreateExecutionEngine(const cinn::ir::Module &module);

  virtual void Compare() {}

  virtual void Reset();

  std::vector<float *> &GetAllDatas() { return all_datas_; }
  int GetOutDims() { return out_dims_; }

 private:
  common::Target target_;
  std::string op_name_;
  float diff_;
  int repeat_;
  std::vector<std::vector<int>> input_shapes_;
  std::vector<std::vector<int>> output_shapes_;
  std::vector<cinn_pod_value_t> all_args_;
  std::vector<float *> all_datas_;
  int out_dims_;
};

}  // namespace tests
}  // namespace cinn

#include "tests/benchmark/test_utils.h"

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace tests {
using ir::Tensor;
auto OpBenchmarkTester::CreateExecutionEngine(const cinn::ir::Module& module) {
  auto engine = cinn::backends::ExecutionEngine::Create({});
  engine->Link(module);
  return engine;
}

void OpBenchmarkTester::TestOp(const std::string& test_name,
                               const hlir::framework::NodeAttr& attrs,
                               const std::vector<Type>& out_types,
                               bool use_default_stragegy) {
  auto module        = CreateCinnModule(attrs, out_types, use_default_stragegy);
  auto engine        = CreateExecutionEngine(module);
  auto test_func_ptr = reinterpret_cast<void (*)(void**, int32_t)>(engine->Lookup(op_name_));
  CreateBuffer();
  LOG(INFO) << "Testing " << test_name;
  cinn::utils::Timer timer;
  timer.Start();
  for (int i = 0; i < repeat_; i++) {
    test_func_ptr(reinterpret_cast<void**>(all_args_.data()), all_args_.size());
  }
  double test_op_time = timer.Stop() / repeat_;
  LOG(INFO) << "repeat times: " << repeat_ << ", kernel run time: " << test_op_time << " ms";
  Compare();
  Reset();
}

Module OpBenchmarkTester::CreateCinnModule(const hlir::framework::NodeAttr& attrs,
                                           const std::vector<Type>& out_types,
                                           bool use_default_stragegy) {
  std::vector<std::vector<Expr>> expr_shapes;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outs;
  std::vector<Tensor> rets;
  poly::StageMap stages;
  std::vector<Expr> output_shape_expr;
  for (int i = 0; i < input_shapes_.size(); i++) {
    std::vector<Expr> expr_shape;
    for (int j = 0; j < input_shapes_[i].size(); ++j) {
      expr_shape.push_back(Expr(input_shapes_[i][j]));
    }
    expr_shapes.push_back(expr_shape);
    Placeholder<float> input(common::UniqName("input"), expr_shape);
    inputs.push_back(input.tensor());
    rets.push_back(input.tensor());
  }
  if (use_default_stragegy) {
    auto strategy = hlir::framework::Operator::GetAttrs<hlir::framework::StrategyFunction>("CINNStrategy");
    auto op       = hlir::framework::Operator::Get(op_name_);
    CHECK(op) << op_name_ << " isn't supported yet\n";
    auto impl = hlir::framework::OpStrategy::SelectImpl(strategy[op](attrs, inputs, out_types, input_shapes_, target_));
    std::vector<common::CINNValue> temp_inputs;
    for (auto& tensor : inputs) {
      temp_inputs.push_back(common::CINNValue(tensor));
    }
    common::CINNValuePack C = impl->fcompute(common::CINNValuePack(temp_inputs));
    stages                  = C.back();
    C                       = impl->fschedule(C);
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      stages->InsertLazily(temp.as_tensor_ref());
      std::vector<Expr> output_shape_expr = temp.as_tensor_ref()->domain_without_reduce_axis();
      std::vector<int> output_shape;
      for (auto& shape : output_shape_expr) {
        output_shape.push_back(shape.as_int32());
      }
      output_shapes_.push_back(output_shape);
    }
    C = impl->fschedule(C);
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      rets.push_back(temp.as_tensor_ref());
    }

  } else {
    stages = CreateStages(inputs);
    outs   = CreateSpecificStrategy(inputs, &stages);

    for (auto& out : outs) {
      stages->InsertLazily(out);
      rets.push_back(out);
      std::vector<Expr> output_shape_expr = out->domain_without_reduce_axis();
      std::vector<int> output_shape;
      for (auto& shape : output_shape_expr) {
        output_shape.push_back(shape.as_int32());
      }
      output_shapes_.push_back(output_shape);
    }
  }
  auto func = Lower(op_name_, stages, rets);
  LOG(INFO) << func;
  Module::Builder builder("module_" + op_name_, target_);
  builder.AddFunction(func);
  return builder.Build();
}

void OpBenchmarkTester::CreateBuffer() {
  std::vector<cinn_pod_value_t> args;
  for (auto& input_shape : input_shapes_) {
    auto* buffer = common::BufferBuilder(Float(32), input_shape).set_align(32).set_random().Build();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
    all_datas_.push_back(reinterpret_cast<float*>(buffer->memory));
  }
  CHECK(!output_shapes_.empty()) << "output shapes shouldn't be empty\n";
  // default only consider the last out as the kernel args
  for (auto& output_shape : output_shapes_) {
    auto* buffer = common::BufferBuilder(Float(32), output_shapes_.back()).set_align(32).set_zero().Build();
    CHECK(buffer);
    out_dims_ = buffer->num_elements();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
    all_datas_.push_back(reinterpret_cast<float*>(buffer->memory));
  }
}

void OpBenchmarkTester::Reset() {
  CHECK(!all_datas_.empty());
  for (int i = 0; i < out_dims_; ++i) {
    all_datas_.back()[i] = 0.f;
  }
}

}  // namespace tests
}  // namespace cinn

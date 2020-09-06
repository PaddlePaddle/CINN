#include "cinn/frontend/syntax.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace frontend {

std::unique_ptr<Program> CreateAddProgram() {
  const int M = 32;
  const int N = 24;

  Placeholder a(Float(32), {M, N});
  Placeholder b(Float(32), {M, N});
  std::unique_ptr<Program> program(new Program);

  auto c = program->add(a, b);
  auto d = program->add(a, c);

  program->SetInputs({a, b});
  program->Validate();

  return program;
}

TEST(syntax, basic) {
  auto program = CreateAddProgram();
  // output program
  for (int i = 0; i < program->size(); i++) {
    LOG(INFO) << "instruction: " << (*program)[i];
  }
}

void SetRandData(hlir::framework::Tensor* tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    unsigned int seed = j;
    data[j]           = (rand_r(&seed) * 1.f) / RAND_MAX;  // All random data
  }
}

TEST(syntax, program_execute_multi_elementwise_add) {
  auto program = CreateAddProgram();
  auto graph   = std::make_shared<hlir::framework::Graph>(*program);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  auto scope    = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("a");
  scope->Var<hlir::framework::Tensor>("b");

  auto A = scope->GetTensor("a");
  auto B = scope->GetTensor("b");
  SetRandData(A, target);
  SetRandData(B, target);

  runtime_program->Execute();
}

TEST(syntax, program_execute_fc) {
  const int B = 10;  // batch size
  const int M = 32;
  const int K = 18;
  const int N = 24;

  Placeholder a(Float(32), {B, M, K});
  Placeholder w(Float(32), {K, N});  // weight
  Placeholder b(Float(32), {N});     // bias

  Program program;
  auto mul_out = program.mul(a, w, false /*trans_a*/, false /*trans_b*/, 2, 1);
  LOG(INFO) << "mul_out: " << mul_out;
  auto add_out = program.elementwise_add(mul_out, b, 1);
  program.SetInputs({a, w, b});

  auto graph = std::make_shared<hlir::framework::Graph>(program);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  auto scope    = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("a");
  scope->Var<hlir::framework::Tensor>("w");
  scope->Var<hlir::framework::Tensor>("b");

  auto at = scope->GetTensor("a");
  auto wt = scope->GetTensor("w");
  auto bt = scope->GetTensor("b");
  SetRandData(at, target);
  SetRandData(wt, target);
  SetRandData(bt, target);
}

}  // namespace frontend
}  // namespace cinn

#include "cinn/frontend/syntax.h"

#include <gtest/gtest.h>

#include <memory>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

using hlir::framework::Scope;
using utils::Join;

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

void SetRandData(const hlir::framework::Tensor& tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    data[j] = (rand() * 1.f) / RAND_MAX;  // All random data
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

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A = scope->GetTensor("A");
  auto B = scope->GetTensor("B");
  SetRandData(A, target);
  SetRandData(B, target);

  runtime_program->Execute();
}

TEST(syntax, program_execute_multi_elementwise_add2) {
  auto program = CreateAddProgram();
  auto graph   = std::make_shared<hlir::framework::Graph>(*program);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  auto scope    = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A = scope->GetTensor("A");
  auto B = scope->GetTensor("B");
  SetRandData(A, target);
  SetRandData(B, target);

  runtime_program->Execute();
}

TEST(syntax, program_execute_fc) {
  const int B = 10;  // batch size
  const int M = 32;
  const int K = 18;
  const int N = 24;

  Placeholder a(Float(32), {B, M, K}, "A");
  Placeholder w(Float(32), {K, N}, "W");  // weight
  Placeholder b(Float(32), {N}, "B");     // bias

  Program program;
  auto mul_out = program.mul(a, w, 2, 1);
  auto add_out = program.add(mul_out, b);
  program.SetInputs({a, w, b});
  program.Validate();

  auto graph = std::make_shared<hlir::framework::Graph>(program);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  auto scope    = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(a.id()));
  scope->Var<hlir::framework::Tensor>(std::string(w.id()));
  scope->Var<hlir::framework::Tensor>(std::string(b.id()));
  scope->Var<hlir::framework::Tensor>(std::string(mul_out->id));

  auto at        = scope->GetTensor(std::string(a.id()));
  auto wt        = scope->GetTensor(std::string(w.id()));
  auto bt        = scope->GetTensor(std::string(b.id()));
  auto fake_outt = scope->GetTensor(std::string(mul_out->id));
  auto add_outt  = scope->GetTensor(std::string(add_out->id));
  SetRandData(at, target);
  SetRandData(wt, target);
  SetRandData(bt, target);

  runtime_program->Execute();
}

// Load a simple Paddle model, execute it
TEST(load_paddle_model, fc_execute) {
  auto scope = std::make_shared<Scope>();

  auto [program, var_map, var_map_paddle_to_program] =
      LoadPaddleProgram(FLAGS_model_dir, scope.get(), false /*is_combined*/);
  var_map["A"]->shape = {1, 30};
  program->SetInputs({var_map["A"]});
  program->Validate();

  LOG(INFO) << "program:\n" << *program;

  auto graph = std::make_shared<hlir::framework::Graph>(*program);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  scope         = BuildScope(target, graph, scope);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  auto at       = scope->GetTensor("A");
  auto* at_data = at->mutable_data<float>(common::DefaultHostTarget());
  for (int i = 0; i < at->shape().numel(); i++) at_data[i] = 1.f;

  runtime_program->Execute();

  LOG(INFO) << "scope.names: " << Join(scope->var_names(), ",");

  const std::string output_name = "fc_0.tmp_1";
  auto tensor                   = scope->GetTensor(var_map_paddle_to_program.at(output_name));
  LOG(INFO) << "tensor.shape: " << utils::Join(tensor->shape().data(), ",");
  auto* data = tensor->data<float>();
  for (int i = 0; i < 10; i++) LOG(INFO) << "data: " << data[i];
}

}  // namespace frontend
}  // namespace cinn

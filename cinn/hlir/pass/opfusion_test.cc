
#include <gtest/gtest.h>

#include <memory>

#include "cinn/cinn.h"
#include "cinn/frontend/syntax.h"
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

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

void SetRandData(const hlir::framework::Tensor& tensor, Target target) {
#ifdef CINN_WITH_CUDA
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (float& v : host_memory) {
    v = (rand() * 1.f) / RAND_MAX;  // All random data
  }
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_memory.data(),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyHostToDevice));
#else
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    data[j] = (rand() * 1.f) / RAND_MAX;  // All random data
  }
#endif
}

// add+relu
TEST(fuse_add_relu, fuse_add_relu) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");
  Placeholder B(Float(32), {64}, "B");

  Program program;
  auto c = program.elementwise_add(A, B, 1);
  auto d = program.relu(c);

  Target target = GetTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);

  runtime_program->Execute();
}

// add+add
TEST(fuse_add, fuse_add) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");
  Placeholder B(Float(32), {64}, "B");
  Placeholder C(Float(32), {64}, "C");

  Program program;
  auto c = program.elementwise_add(A, B, 1);
  auto d = program.elementwise_add(c, C, 1);

  Target target = GetTarget();
  program.SetInputs({A, B, C});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);

  runtime_program->Execute();
}

// conv+bn+add+add+relu
TEST(conv_bn_conv, conv_bn_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {1, 64, 1, 1}, "D");
  Placeholder E(Float(32), {1, 64, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["stride"]        = std::vector<int>({2, 2});
  attrs["dilation"]      = std::vector<int>({1, 1});
  attrs["padding"]       = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"]   = src_layout;

  std::unordered_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.batchnorm(c, Scale, Bias, Mean, Variance, attrs1);
  auto e = program.elementwise_add(d, C);
  auto f = program.elementwise_mul(e, D);
  auto g = program.relu(f);

  Target target = GetTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");
  scope->Var<hlir::framework::Tensor>("E");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  auto E1 = scope->GetTensor("E");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);
  SetRandData(D1, target);
  SetRandData(E1, target);

  runtime_program->Execute();
}

// conv+add
TEST(fuse_conv_add, fuse_conv_add) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["stride"]        = std::vector<int>({2, 2});
  attrs["dilation"]      = std::vector<int>({1, 1});
  attrs["padding"]       = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"]   = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C, 1);

  Target target = GetTarget();
  program.SetInputs({A, B, C});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);
  SetRandData(D1, target);

  runtime_program->Execute();
}

// conv+add+mul
TEST(conv_add_mul, conv_add_mul) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Placeholder Scale(Float(32), {1, 64, 1, 1}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["stride"]        = std::vector<int>({2, 2});
  attrs["dilation"]      = std::vector<int>({1, 1});
  attrs["padding"]       = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"]   = src_layout;

  std::unordered_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, Scale);
  auto e = program.elementwise_mul(d, Bias, 1);

  Target target = GetTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);
  SetRandData(D1, target);

  runtime_program->Execute();
}

/**
 *  complex case: diamond structure
 *         conv
 *        /     \
 *      add    relu
 *        \     /
 *          add
 */
TEST(complex1, complex1) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {1, 64, 1, 1}, "D");
  Placeholder E(Float(32), {1, 64, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["stride"]        = std::vector<int>({2, 2});
  attrs["dilation"]      = std::vector<int>({1, 1});
  attrs["padding"]       = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"]   = src_layout;

  std::unordered_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C);
  auto e = program.relu(c);
  auto f = program.elementwise_add(d, e);

  Target target = GetTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);

  runtime_program->Execute();
}

TEST(complex2, complex2) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {3, 1, 7, 7}, "B");
  Placeholder C(Float(32), {1, 3, 112, 112}, "C");
  Placeholder D(Float(32), {1, 3, 1, 1}, "D");
  Placeholder E(Float(32), {1, 3, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["stride"]        = std::vector<int>({2, 2});
  attrs["dilation"]      = std::vector<int>({1, 1});
  attrs["padding"]       = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"]   = src_layout;

  std::unordered_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.depthwise_conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C);
  auto e = program.relu(c);
  auto f = program.elementwise_add(d, e);

  Target target = GetTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData(A1, target);
  SetRandData(B1, target);
  SetRandData(C1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn

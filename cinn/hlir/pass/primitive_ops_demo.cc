
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

// batch_norm primitives
TEST(batch_norm_meta, batch_norm_meta) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "C");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  attrs["epsilon"] = static_cast<float>(0.001);

  auto a = program.batchnorm(A, Scale, Bias, Mean, Variance, attrs);

  auto b = program.fused_batchnorm_inference(A, Scale, Bias, Mean, Variance, attrs);

  Target target = GetTarget();
  program.SetInputs({A});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
#ifndef CINN_WITH_CUDA
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
#endif
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");

  auto A1 = scope->GetTensor("A");
  SetRandData(A1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn

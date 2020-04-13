#include "cinn/backends/codegen_cuda_dev.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"

namespace cinn {
namespace backends {

TEST(CodeGenCUDA, basic) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", {A, B, C});

  auto compiled = codegen.Compile(func);

  std::cout << compiled << std::endl;
}

TEST(CodeGenCUDA, Module) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", {A, B, C});

  Module module("module", target);
  module.Append(func);

  Outputs outputs;
  outputs = outputs.cuda_source("generated1.cu");
  codegen.Compile(module, outputs);
}

}  // namespace backends
}  // namespace cinn

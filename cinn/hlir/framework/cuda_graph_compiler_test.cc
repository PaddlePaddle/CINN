#include <gtest/gtest.h>
#include <stdlib.h>

#include <any>
#include <string>
#include <tuple>
#include <vector>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/common/cuda_test_helper.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/packed_func.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/use_extern_funcs.h"

namespace cinn {
namespace hlir {
namespace framework {

Tensor GetTensor(const std::shared_ptr<Scope>& scope, const std::string& name) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = std::get<Tensor>(*var);
  return tensor;
}

void CudaSetRandData(const Tensor& tensor, const Target& target) {
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (float& v : host_memory) {
    v = (rand() * 1.f) / RAND_MAX;  // All random data
  }
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_memory.data(),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

std::vector<float> CudaGetData(const Tensor& tensor, const Target& target) {
  auto* A_data = tensor->mutable_data<float>(target);
  std::vector<float> host_data(tensor->shape().numel(), 0);

  CUDA_CALL(cudaMemcpy(host_data.data(),
                       reinterpret_cast<void*>(A_data),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return host_data;
}

TEST(GraphCompiler, RunModel) {
  using attr_t = hlir::framework::AttrType;
  frontend::Program prog;
  // TODO(Superjomn) Replace with Placeholder here.
  Expr M(100);
  Expr N(32);
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t   = Float(32);
  a->shape = {M.as_int32(), N.as_int32()};
  b->shape = {M.as_int32(), N.as_int32()};
  a->type  = t;
  b->type  = t;
  auto c   = prog.add(a, b);
  auto d   = prog.add(c, b);
  auto e   = prog.add(c, d);
  std::unordered_map<std::string, attr_t> attr_store;
  attr_store["scale"] = 1.0f;
  attr_store["bias"]  = 0.0f;
  auto o              = prog.scale(e, attr_store);
  ASSERT_EQ(prog.size(), 4UL);
  auto g = std::make_shared<Graph>(prog);
  ApplyPass(g.get(), "InferShape");

  Target target(Target::OS::Linux, Target::Arch::NVGPU, Target::Bit::k64, {});
  auto scope = BuildScope(target, g);

  GraphCompiler gc(target, scope, g);
  std::unique_ptr<Program> program = gc.Build();

  auto A = GetTensor(scope, "A");
  auto B = GetTensor(scope, "B");
  CudaSetRandData(A, target);
  CudaSetRandData(B, target);

  program->Execute();
  auto host_data1 = CudaGetData(A, target);
  auto host_data2 = CudaGetData(B, target);
  auto Out        = GetTensor(scope, o->id);
  auto host_data3 = CudaGetData(Out, target);

  for (int i = 0; i < Out->shape().numel(); i++) {
    LOG_FIRST_N(INFO, 10) << "data[" << i << "]: "
                          << "2 * " << host_data1[i] << " + "
                          << "3 * " << host_data2[i] << " = " << host_data3[i];
    EXPECT_NEAR(host_data3[i], 2 * host_data1[i] + 3 * host_data2[i], 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn

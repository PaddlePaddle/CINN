#include "cinn/hlir/framework/instruction.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/common/test_helper.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(int M, int N) {
  Expr m(M);
  Expr n(N);

  Placeholder<float> x("x", {m, n});
  Placeholder<float> y("y", {m, n});

  auto z = Compute({m, n}, [=](Expr i, Expr j) { return x(i, j) + y(i, j); }, "z");

  auto stages = CreateStages({z});
  auto fn     = Lower("fn", stages, {x, y, z});

  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

void InstantiateScope(int M, int N, Scope* scope) {
  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto&& tensor = scope->GetTensor(name);
    tensor->Resize(Shape{{M, N}});
    auto* data = tensor->mutable_data<float>(common::DefaultHostTarget());
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
  }
}

TEST(Instruction, basic) {
  const int M = 10;
  const int N = 20;

  Scope scope;
  InstantiateScope(M, N, &scope);
  // create Instruction
  Instruction instr(common::DefaultHostTarget(), &scope, {"x", "y"}, {"z"});
  auto jit     = GetLoweredFunc(M, N);
  auto fn_addr = jit->Lookup("fn");
  CHECK(fn_addr);

  instr.SetLoweredFunc(reinterpret_cast<lower_func_ptr_t>(fn_addr));
  instr.Run();

  // check result
  {
    auto* xd = scope.GetTensor("x")->data<float>();
    auto* yd = scope.GetTensor("y")->data<float>();
    auto* zd = scope.GetTensor("z")->data<float>();

    for (int i = 0; i < M * N; i++) {
      LOG_FIRST_N(INFO, 3) << "data: " << xd[i] << " + " << yd[i] << " = " << zd[i];
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  }
}

TEST(Instruction, RunWithRawPodArgs) {
  const int M       = 10;
  const int N       = 20;
  const auto& shape = Shape({{M, N}});

  std::map<std::string, cinn_pod_value_t> name2podargs;
  // case 1: create cinn_pod_value_t arguments dicrectly
  std::vector<cinn_buffer_t> args_buffer;  // store the buffer objects
  auto* default_memory_mng = MemoryManager::Global().RetrieveSafely(common::DefaultHostTarget().arch);

  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    args_buffer.emplace_back();
    auto& buffer = args_buffer.back();
    buffer.resize(reinterpret_cast<const cinn_dimension_t*>(shape.data().data()), shape.size());
    buffer.memory = reinterpret_cast<uint8_t*>(default_memory_mng->malloc(shape.numel() * sizeof(float)));
    auto* data    = buffer.memory;
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
    name2podargs.emplace(name, &buffer);
  }

  // create Instruction
  auto jit     = GetLoweredFunc(M, N);
  auto fn_addr = jit->Lookup("fn");
  CHECK(fn_addr);

  Instruction instr(common::DefaultHostTarget(), nullptr, {"x", "y"}, {"z"});  // empty scope
  instr.SetLoweredFunc(reinterpret_cast<lower_func_ptr_t>(fn_addr));
  instr.Run(&name2podargs);  // run with a arguments map passed

  auto check_equal_by_element = [&name2podargs]() {
    auto xd = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at("x"))->memory);
    auto yd = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at("y"))->memory);
    auto zd = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at("z"))->memory);
    for (int i = 0; i < M * N; ++i) {
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  };

  // check instruction run correctly
  check_equal_by_element();

  // case 2: create cinn_pod_value_t arguments from scope;
  Scope scope;
  InstantiateScope(M, N, &scope);
  name2podargs.clear();

  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto&& tensor = scope.GetTensor(name);
    name2podargs.emplace(name, tensor->buffer());
  }
  instr.Run(&name2podargs);
  // check instruction run correctly
  check_equal_by_element();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

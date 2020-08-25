#include "cinn/hlir/framework/instruction.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/llvm/simple_jit.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(int M, int N) {
  Expr m(M);
  Expr n(N);

  Placeholder<float> x("x", {m, n});
  Placeholder<float> y("y", {m, n});

  auto z = Compute(
      {m, n}, [=](Expr i, Expr j) { return x(i, j) + y(i, j); }, "z");

  auto stages = CreateStages({z});
  auto fn     = Lower("fn", stages, {x, y, z});

  lang::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

TEST(Instruction, basic) {
  const int M = 10;
  const int N = 20;

  Scope scope;

  auto get_tensor = [&](const std::string& name) {
    auto* var    = scope.Var<Tensor>(name);
    auto& tensor = std::get<Tensor>(*var);
    return tensor;
  };

  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto tensor = get_tensor(name);
    tensor.Resize(Shape{{M, N}});
    auto* data = tensor.mutable_data<float>(common::DefaultHostTarget());
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
  }

  // create Instruction
  Instruction instr(common::DefaultHostTarget(), &scope, {"x", "y"}, {"z"});
  auto jit     = GetLoweredFunc(M, N);
  auto fn_addr = jit->Lookup("fn");
  CHECK(fn_addr);

  instr.SetLoweredFunc(reinterpret_cast<lower_func_ptr_t>(fn_addr));
  instr.Run();

  // check result
  {
    auto xd = get_tensor("x").data<float>();
    auto yd = get_tensor("y").data<float>();
    auto zd = get_tensor("z").data<float>();

    for (int i = 0; i < M * N; i++) {
      LOG_FIRST_N(INFO, 3) << "data: " << xd[i] << " + " << yd[i] << " = " << zd[i];
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

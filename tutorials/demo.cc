#include <iostream>

#include "cinn/backends/compiler.h"
#include "cinn/cinn.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace backends {

// test x86 compiler
int run() {
  Expr M(4), N(4);

  auto create_module = [&]() {
    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [=](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");
    return std::make_tuple(A, B, C);
  };

  // test x86
  auto [A, B, C] = create_module();  // NOLINT

  auto stages = CreateStages({C});

  auto fn = Lower("fn", stages, {A, B, C});

  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto compiler = Compiler::Create(common::DefaultHostTarget());
  compiler->Build(builder.Build());

  auto* fnp = compiler->Lookup("fn");
  if (fnp == nullptr) {
    std::cerr << "lookup function failed." << std::endl;
    return 1;
  }

  auto* Ab = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* Bb = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* Cb = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto args = common::ArgsBuilder().Add(Ab).Add(Bb).Add(Cb).Build();
  fnp(args.data(), args.size());

  // test result
  auto* Ad = reinterpret_cast<float*>(Ab->memory);
  auto* Bd = reinterpret_cast<float*>(Bb->memory);
  auto* Cd = reinterpret_cast<float*>(Cb->memory);
  for (int i = 0; i < Ab->num_elements(); i++) {
    if (abs(Ad[i] + Bd[i] - Cd[i]) > 1e-5) {
      std::cerr << "ERROR: Compute failed." << std::endl;
      return 1;
    }
  }

  std::cout << "run demo successfully." << std::endl;
  return 0;
}
}  // namespace backends
}  // namespace cinn

int main() { return cinn::backends::run(); }

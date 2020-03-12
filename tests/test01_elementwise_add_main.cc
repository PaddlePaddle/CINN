#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {

TEST(test01_elementwise_add, basic) {
  Placeholder<float> A("A", {100, 20});
  Placeholder<float> B("B", {100, 20});

  Buffer C_buf(Float(32));
  auto C = Compute(
      {100, 20}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module module("module1", target);

  auto funcs = Lower("add1", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  module.Append(C_buf);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add.h").c_source("./test01_elementwise_add.cc");
  compiler.Compile(module, outputs);
}

}  // namespace cinn

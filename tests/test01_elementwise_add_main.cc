#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {

TEST(test01_elementwise_add, basic) {
  Placeholder<float> A("A", {100, 32});
  Placeholder<float> B("B", {100, 32});

  Buffer C_buf(Float(32));
  auto C = Compute(
      {100, 32}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module module("module1", target);

  auto funcs = Lower("add1", {A, B, C});

  auto func = Optimize(funcs);
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(C_buf);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add.h").c_source("./test01_elementwise_add.cc");
  compiler.Compile(module, outputs);
}

TEST(test01_elementwise_add, vectorize) {
  Placeholder<float> A("A", {100, 32});
  Placeholder<float> B("B", {100, 32});

  Buffer C_buf(Float(32));
  auto C = Compute(
      {100, 32}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);
  C->stage()->Vectorize(1, 8);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module module("module2", target);

  auto funcs = Lower("add1_vectorize", {A, B, C});

  auto func = Optimize(funcs);
  LOG(INFO) << "after optim:\n" << func;
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(C_buf);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_vectorize.h").c_source("./test01_elementwise_add_vectorize.cc");
  compiler.Compile(module, outputs);
}

}  // namespace cinn

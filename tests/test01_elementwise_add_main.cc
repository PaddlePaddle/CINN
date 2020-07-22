#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {

TEST(test01_elementwise_add, basic) {
  Expr M(100), N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Buffer C_buf(Float(32));
  // auto C = Compute(
  //    {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  auto C = primitive::add(A, B, "C");
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module::Builder builder("module1", target);

  auto func = Lower("add1", {A, B, C});

  builder.AddFunction(func);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add.h").c_source("./test01_elementwise_add.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(test01_elementwise_add, vectorize) {
  Expr M(100);
  Expr N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Buffer C_buf(Float(32));
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);
  C->stage()->Vectorize(1, 8);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module::Builder builder("module2", target);

  auto funcs = Lower("add1_vectorize", {A, B, C});

  auto func = Optimize(funcs);
  LOG(INFO) << "after optim:\n" << func;
  builder.AddFunction(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(C_buf);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_vectorize.h").c_source("./test01_elementwise_add_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn

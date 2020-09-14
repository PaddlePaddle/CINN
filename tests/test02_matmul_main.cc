#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"
#include "tests/test02_helper.h"

namespace cinn {
using poly::Iterator;

Expr M(1024);
Expr N(1024);
Expr K(1024);

TEST(test02_matmul, basic) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  auto stages = CreateStages({C_init, C});

  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  {
    Module::Builder builder("module1", target);
    auto func = Lower("matmul", stages, {A, B, C});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul.h").c_source("./test02_matmul.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  // Tile
  {
    stages[C]->Tile(0, 1, 4, 4);

    Module::Builder builder("module2", target);
    auto func = Lower("matmul_tile", stages, {A, B, C});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_tile.h").c_source("./test02_matmul_tile.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(matmul, Split) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  auto stages = CreateStages({C_init, C});

  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  auto [i0, i1] = stages[C]->Split(2, 16);
  std::vector<Iterator> iterators(
      {stages[C]->ith_iterator(1), stages[C]->ith_iterator(0), stages[C]->ith_iterator(2), stages[C]->ith_iterator(3)});
  stages[C]->Reorder(iterators);

  Module::Builder builder("module3", target);
  auto func = Lower("matmul_split", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_split.h").c_source("./test02_matmul_split.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Blocking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  auto stages = CreateStages({C_init, C});
  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  // Blocking by loop tiling.
  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);
    stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
  }

  Module::Builder builder("module_block", target);
  auto func = Lower("matmul_block", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_block.h").c_source("./test02_matmul_block.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Vectorization) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  auto stages = CreateStages({C_init, C});
  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  // Blocking by loop tiling.
  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);
    stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_vectorize", target);
  auto func = Lower("matmul_vectorize", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_vectorize.h").c_source("./test02_matmul_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, LoopPermutation) {
  auto module = tests::CreateMatmulLoopPermutation(common::DefaultHostTarget(), 1024, 1024, 1024);

  CodeGenCX86 compiler(common::DefaultHostTarget(), CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_loop_permutation.h").c_source("./test02_matmul_loop_permutation.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, ArrayPacking) {
  auto target = common::DefaultHostTarget();

  auto module = tests::CreateMatmulArrayPacking(target, 1024, 1024, 1024);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing.h").c_source("./test02_matmul_array_packing.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, varient_shape) {
  Var M("M");  // M is a symbol.
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  auto stages = CreateStages({C_init, C});
  stages[C]->CtrlDepend(C_init);
  stages[C_init]->ShareBufferWith(stages[C]);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  {
    Module::Builder builder("matmul_dynamic_shape", target);
    auto func = Lower("matmul_dynamic_shape", stages, {A, B, C}, {M});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_varient_shape.h").c_source("./test02_matmul_varient_shape.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  {
    int bn                                    = 32;
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);  // NOLINT

    Module::Builder builder("matmul_dynamic_shape_tile", target);
    auto func = Lower("matmul_dynamic_shape_tile", stages, {A, B, C} /*tensors*/, {M} /*scalars*/);
    LOG(INFO) << "func " << Expr(func);

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;

    outputs =
        outputs.c_header("./test02_matmul_varient_shape_tile.h").c_source("./test02_matmul_varient_shape_tile.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(matmul, ArrayPacking_dynamic_shape) {
  Var M("M");
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  Expr bn(32);

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "packedB");

  auto C = Compute({M, N}, [&](Expr i, Expr j) { return Sum(A(i, k) * packedB(j / bn, k, j % bn)); }, "C", {k});

  auto stages = CreateStages({C_init, C});
  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);

  LOG(INFO) << "stage: " << stages[packedB]->transformed_domain();
  stages[packedB]->Vectorize(2, 8);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn.as_int32(), bn.as_int32());
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing_dynamic_shape", target);
  auto func = Lower("matmul_array_packing_dynamic_shape", stages, {A, B, C}, {M}, {packedB}, &builder);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing_dynamic_shape.h")
                .c_source("./test02_matmul_array_packing_dynamic_shape.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, call) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");
  Buffer C_buf(Float(32));

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  auto stages = CreateStages({C_init, C});
  stages[C]->CtrlDepend(C_init);

  Module::Builder builder("module_call", target);
  {
    stages[C]->ShareBufferWith(stages[C_init]);
    auto func = Lower("matmul_kernel", stages, {A, B, C});

    builder.AddFunction(func);
  }

  {  // main
    std::vector<lang::ReturnType> returns({lang::ReturnType{Float(32), C->shape, C->name}});
    auto tensors = lang::CallLowered("matmul_kernel", {A, B}, returns);
    auto C       = tensors[0];

    LOG(INFO) << "stage domain: " << stages[C]->domain();
    auto fn = Lower("matmul_main", stages, {A, B, C}, {});
    builder.AddFunction(fn);
  }

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_call.h").c_source("./test02_matmul_call.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn

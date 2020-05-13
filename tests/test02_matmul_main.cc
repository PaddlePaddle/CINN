#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {
using poly::Iterator;

Expr M(1024);
Expr N(1024);
Expr K(1024);

TEST(test02_matmul, basic) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");
  Buffer C_buf(Float(32));

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Module::Builder builder("module1", target);
    auto func = Lower("matmul", {A, B, C, C_init});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul.h").c_source("./test02_matmul.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  // Tile
  {
    C->stage()->Tile(0, 1, 4, 4);

    Module::Builder builder("module2", target);
    auto func = Lower("matmul_tile", {A, B, C, C_init});

    builder.AddFunction(func);
    // module.Append(C_buf);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_tile.h").c_source("./test02_matmul_tile.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(matmul, Split) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Iterator i0, i1;
  std::tie(i0, i1) = C->stage()->Split(2, 16);
  std::vector<Iterator> iterators({C->stage()->ith_iterator(1),
                                   C->stage()->ith_iterator(0),
                                   C->stage()->ith_iterator(2),
                                   C->stage()->ith_iterator(3)});
  C->stage()->Reorder(iterators);

  Module::Builder builder("module3", target);
  auto func = Lower("matmul_split", {A, B, C, C_init});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_split.h").c_source("./test02_matmul_split.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Blocking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);

  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    Iterator k_outer, k_inner;
    std::tie(k_outer, k_inner) = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
  }

  Module::Builder builder("module_block", target);
  auto func = Lower("matmul_block", {A, B, C, C_init});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_block.h").c_source("./test02_matmul_block.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Vectorization) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  // ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    Iterator k_outer, k_inner;
    std::tie(k_outer, k_inner) = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

    C->stage()->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_vectorize", target);
  auto func = Lower("matmul_vectorize", {A, B, C, C_init});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_vectorize.h").c_source("./test02_matmul_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, LoopPermutation) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    Iterator k_outer, k_inner;

    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split("k", 4);

    C_init->stage()->Vectorize(1, 8);
    C_init->stage()->Unroll(1);

    C->stage()->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});

    C->stage()->Vectorize(j_inner, 8);
    C->stage()->Unroll(5);
  }

  Module::Builder builder("module_loop_permutation", target);
  auto func = Lower("matmul_loop_permutation", {A, B, C, C_init});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_loop_permutation.h").c_source("./test02_matmul_loop_permutation.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, ArrayPacking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  Expr bn(32);

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "packedB");
  packedB->WithBuffer();
  LOG(INFO) << "stage: " << packedB->stage()->transformed_domain();
  packedB->stage()->Vectorize(2, 8);

  auto C = Compute({M, N}, [&](Expr i, Expr j) { return Sum(A(i, k) * packedB(j / bn, k, j % bn)); }, "C", {k});
  C->Bind(C_init->buffer);

  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    Iterator k_outer, k_inner;

    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    std::tie(k_outer, k_inner)                   = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    C->stage()->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing", target);
  auto func = Lower("matmul_array_packing", {A, B, C, C_init, packedB});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing.h").c_source("./test02_matmul_array_packing.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, varient_shape) {
  Var M("M");  // M is a symbol.
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Module::Builder builder("matmul_dynamic_shape", target);
    auto func = Lower("matmul_dynamic_shape", {A, B, C, C_init}, {M});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_varient_shape.h").c_source("./test02_matmul_varient_shape.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  {
    int bn                                    = 32;
    auto [i_outer, i_inner, j_outer, j_inner] = C->stage()->Tile(0, 1, bn, bn);  // NOLINT

    Module::Builder builder("matmul_dynamic_shape_tile", target);
    auto func = Lower("matmul_dynamic_shape_tile", {A, B, C, C_init} /*tensors*/, {M} /*scalars*/);
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

  Var k(K.as_int32(), "k");

  Expr bn(32);

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "packedB");
  packedB->WithBuffer();
  LOG(INFO) << "stage: " << packedB->stage()->transformed_domain();
  packedB->stage()->Vectorize(2, 8);

  auto C = Compute({M, N}, [&](Expr i, Expr j) { return Sum(A(i, k) * packedB(j / bn, k, j % bn)); }, "C", {k});
  C->Bind(C_init->buffer);

  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    Iterator k_outer, k_inner;

    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    std::tie(k_outer, k_inner)                   = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    C->stage()->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing_dynamic_shape", target);
  auto func = Lower("matmul_array_packing_dynamic_shape", {A, B, C, C_init}, {M}, {packedB}, &builder);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing_dynamic_shape.h")
                .c_source("./test02_matmul_array_packing_dynamic_shape.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, call) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k");
  Buffer C_buf(Float(32));

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Module::Builder builder("module_call", target);
  {
    auto func = Lower("matmul_kernel", {A, B, C, C_init});

    builder.AddFunction(func);
  }

  {  // main
    std::vector<lang::ReturnType> returns({lang::ReturnType{Float(32), C->shape, C->name}});
    auto tensors = lang::Call("matmul_kernel", {A, B}, returns);
    auto C       = tensors[0];

    auto fn = Lower("matmul_main", {A, B, C}, {});
    builder.AddFunction(fn);
  }

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_call.h").c_source("./test02_matmul_call.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn

#include "cinn/optim/transform_polyfor_to_for.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"

namespace cinn {
namespace optim {

TEST(Expr, basic) {
  using namespace ir;  // NOLINT

  Expr M(512);
  Expr K(200);
  Expr N(500);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));
  Var k(K.as_int32(), "k");

  Tensor C = Compute({M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_buf);

  {
    C->stage()->Split("i", 8, poly::SplitRestStrategy::kAuto);
    C->stage()->Split("j", 8, poly::SplitRestStrategy::kAuto);
  }

  // Code gen
  auto funcs = Lower("matmul", {A, B, C});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    lang::Module module("module1", target);
    module.Append(funcs);
    module.Append(C_buf);

    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
    std::cout << "out:\n" << out;
  }

  optim::TransformPolyForToFor(&funcs->body);

  {
    lang::Module module("module1", target);
    module.Append(funcs);
    module.Append(C_buf);

    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
    std::cout << "out:\n" << out;

    auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 512, 500 });
void matmul(void* _args, int32_t num_args)
{
  cinn_buffer_t** args = (cinn_buffer_t**)(_args);
  const cinn_buffer_t* _A = args[0];
  const cinn_buffer_t* _B = args[1];
  cinn_buffer_t* _C = args[2];
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(_A->host_memory);
  const float* B = (const float*)(_B->host_memory);
  float* C = (float*)(_C->host_memory);
  for (int32_t i_outer = 0; i_outer < 64; i_outer += 1) {
    for (int32_t i_inner = 0; i_inner < 8; i_inner += 1) {
      for (int32_t j_outer = 0; j_outer < 62; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < 8; j_inner += 1) {
          for (int32_t k = 0; k < 200; k += 1) {
            C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((1600 * i_outer) + k))] * B[((8 * j_outer) + ((500 * k) + j_inner))]));
          };
        };
      };
      for (int32_t j_outer = 62; j_outer < 63; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-8 * j_outer)); j_inner += 1) {
          for (int32_t k = 0; k < 200; k += 1) {
            C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((1600 * i_outer) + k))] * B[((8 * j_outer) + ((500 * k) + j_inner))]));
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
    EXPECT_EQ(utils::Trim(target_out), utils::Trim(out));
  }
}

}  // namespace optim
}  // namespace cinn

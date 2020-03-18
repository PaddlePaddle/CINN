#include "cinn/optim/vectorize_loops.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {

ir::LoweredFunc CreateFunc() {
  using namespace ir;  // NOLINT

  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));
  Var k(K, "k");

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k"), 8);
    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    C->stage()->Split(j_inner, 8, poly::SplitRestStrategy::kSeparate);
  }

  // Code gen
  auto funcs = Lower("matmul", {A, B, C});
  CHECK_EQ(funcs.size(), 1UL);
  return funcs.front();
}

TEST(VectorizeLoops, Split_separate) {
  using namespace ir;  // NOLINT

  ir::LoweredFunc func = CreateFunc();

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  lang::Module module("module1", target);
  module.Append(func);

  CodeGenC codegen(target);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void matmul(const struct cinn_buffer_t *_A, const struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(cinn_buffer_get_data_const_handle(_A));
  const float* B = (const float*)(cinn_buffer_get_data_const_handle(_B));
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  for (int32_t i_outer = 0; (i_outer <= 3); i_outer += 1) {
    for (int32_t j_outer = 0; (j_outer <= 15); j_outer += 1) {
      for (int32_t k_outer = 0; (k_outer <= 24); k_outer += 1) {
        for (int32_t k_inner = 0; (k_inner <= 7); k_inner += 1) {
          for (int32_t i_inner = 0; (i_inner <= min(31, ((-32 * i_outer) + 99))); i_inner += 1) {
            for (int32_t j_inner_outer = 0; (j_inner_outer <= min(3, ((-4 * j_outer) + 62))); j_inner_outer += 1) {
              if ((7 <= (((-32 * j_outer) - (8 * j_inner_outer)) + 499))) {
                for (int32_t j_inner_inner = 0; (j_inner_inner <= 7); j_inner_inner += 1) {
                  C[((((32 * i_outer) + i_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))] = (C[((((32 * i_outer) + i_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))] + (A[((((32 * i_outer) + i_inner) * 200) + ((8 * k_outer) + k_inner))] * B[((((8 * k_outer) + k_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))]));
                }
              } else {
                for (int32_t j_inner_inner = 0; (j_inner_inner <= (((-32 * j_outer) - (8 * j_inner_outer)) + 499)); j_inner_inner += 1) {
                  C[((((32 * i_outer) + i_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))] = (C[((((32 * i_outer) + i_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))] + (A[((((32 * i_outer) + i_inner) * 200) + ((8 * k_outer) + k_inner))] * B[((((8 * k_outer) + k_inner) * 500) + (((32 * j_outer) + (8 * j_inner_outer)) + j_inner_inner))]));
                }
              };
            };
          };
        };
      };
    };
  };
}
  )ROC";

  EXPECT_EQ(utils::Trim(target_out), utils::Trim(out));
}

TEST(Vectorize, replace_var) {
  using namespace ir;  // NOLINT

  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->Bind(C_buf);

  C->stage()->Vectorize(1, 16);

  auto funcs = Lower("matmul", {A, B, C});
  CHECK_EQ(funcs.size(), 1UL);

  detail::Vectorize(ir::_Var_::Make("j_inner", Int(32)), 16, &funcs.front()->body);

  std::cout << "\n" << funcs.front()->body << std::endl;

  auto target_out = R"ROC(
{
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j_outer <= 31), 1)
    {
      if ((15 <= ((-16 * j_outer) + 499))) {
        poly_for (0, (j_inner <= 15), 1)
        {
          C[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)] = (A[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)] * B[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)])
        }
      } else {
        poly_for (0, (j_inner <= ((-16 * j_outer) + 499)), 1)
        {
          C[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)] = (A[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)] * B[Ramp(((i * 500) + ((16 * j_outer) + 0)),1,16)])
        }
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(target_out), utils::Trim(utils::GetStreamCnt(funcs.front()->body)));

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  lang::Module module("module1", target);
  module.Append(funcs[0]);

  CodeGenC codegen(target);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;
}

}  // namespace optim
}  // namespace cinn

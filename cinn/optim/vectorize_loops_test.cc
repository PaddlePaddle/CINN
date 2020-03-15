#include "cinn/optim/vectorize_loops.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace optim {

TEST(VectorizeLoops, Split_separate) {
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
  ASSERT_EQ(funcs.size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  optim::VectorizeLoops(&funcs[0]->body, target);

  lang::Module module("module1", target);
  module.Append(funcs.front());
  module.Append(C_buf);

  CodeGenC codegen(target);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 });
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

}  // namespace optim
}  // namespace cinn

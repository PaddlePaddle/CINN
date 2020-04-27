#include "hlir/instruction/primitive/dot.h"
#include <gtest/gtest.h>
#include "cinn/cinn.h"

namespace hlir {
namespace instruction {
namespace primitive {
using namespace cinn;  // NOLINT

TEST(Dot, matrix_dot_matrix) {
  Var N("N");

  // Matrix [N x 10] DOT Matrix [10 X 100]
  std::vector<Expr> A_shape({N, Expr(10)});
  std::vector<Expr> B_shape({Expr(10), Expr(100)});
  cinn::Placeholder<float> A("A", A_shape);
  cinn::Placeholder<float> B("B", B_shape);

  Context context;
  DotImpl dotter(&context);
  auto C = dotter(A, B, "C");

  LOG(INFO) << C->body();

  CHECK_EQ(C->shape[0], A_shape[0]);
  CHECK_EQ(C->shape[1], B_shape[1]);

  std::cerr << "Lowered:\n" << Lower("func", {A, B, C}, {}) << std::endl;
}

TEST(Dot, matrix_dot_vector) {
  Var N("N");

  // Matrix [N x 10] DOT Matrix [10 X 100]
  std::vector<Expr> A_shape({N, Expr(10)});
  std::vector<Expr> B_shape({Expr(10)});
  cinn::Placeholder<float> A("A", A_shape);
  cinn::Placeholder<float> B("B", B_shape);

  Context context;
  DotImpl dotter(&context);
  auto C = dotter(A, B, "C");

  LOG(INFO) << C->body();

  CHECK_EQ(C->shape[0], A_shape[0]);
  CHECK_EQ(C->shape.size(), 1UL);
  std::cerr << "Lowered:\n" << Lower("func", {A, B, C}, {}) << std::endl;
}

TEST(Dot, vector_dot_vector) {
  Var N("N");

  // Matrix [N x 10] DOT Matrix [10 X 100]
  std::vector<Expr> A_shape({N});
  std::vector<Expr> B_shape({N});
  cinn::Placeholder<float> A("A", A_shape);
  cinn::Placeholder<float> B("B", B_shape);

  Context context;
  DotImpl dotter(&context);
  auto C = dotter(A, B, "C");

  LOG(INFO) << C->body();

  CHECK_EQ(C->shape[0], Expr(1));
  std::cerr << "Lowered:\n" << Lower("func", {A, B, C}, {}) << std::endl;
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir

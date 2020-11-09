#include "tests/benchmark/test_matmul.h"

#include <gtest/gtest.h>

namespace cinn {
namespace tests {

// default
std::vector<ir::Tensor> MatmulTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                             poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto out = hlir::pe::Matmul(inputs[0], inputs[1]);
  outs.push_back(out);
  (*stages)->InsertLazily(out);
  return outs;
}

// tile
std::vector<ir::Tensor> MatmulTileTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                 poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto out = hlir::pe::Matmul(inputs[0], inputs[1]);
  outs.push_back(out);
  (*stages)->InsertLazily(out);
  (*stages)[out]->Tile(0, 1, 4, 4);
  return outs;
}

// split
std::vector<ir::Tensor> MatmulSplitTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                  poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto out = hlir::pe::Matmul(inputs[0], inputs[1]);
  outs.push_back(out);
  (*stages)->InsertLazily(out);

  auto c_poly_iterators = [&](auto &&... args) {
    std::vector<poly::Iterator> iters;
    (iters.push_back((*stages)[out]->ith_iterator(args)), ...);
    return iters;
  };
  (*stages)[out]->Split(2, 16);
  (*stages)[out]->Reorder(c_poly_iterators(1, 0, 2, 3));

  return outs;
}

// block
std::vector<ir::Tensor> MatmulBlockTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                  poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto k1 = Var(input_shapes_[0][1], "k1");
  CHECK_EQ(input_shapes_.size(), 2U) << "matmul's input shape should be 2.\n";
  CHECK_EQ(input_shapes_[0].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[1].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[0][1], input_shapes_[1][0]) << "matmul's reduce axis shape should be same\n";
  auto C = Compute(
      {Expr(input_shapes_[0][0]), Expr(input_shapes_[1][1])},
      [&](Var i, Var j) { return ReduceSum(inputs[0](i, k1) * inputs[1](k1, j), {k1}); },
      "C");
  (*stages)->InsertLazily(C);
  int bn                                    = 32;
  auto [i_outer, i_inner, j_outer, j_inner] = (*stages)[C]->Tile(0, 1, bn, bn);  // NOLINT
  auto [k_outer, k_inner]                   = (*stages)[C]->Split(k1->name, 4);  // NOLINT
  (*stages)[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

  outs.push_back(C);
  return outs;
}

// vectorize
std::vector<ir::Tensor> MatmulVectorizeTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                      poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto k1 = Var(input_shapes_[0][1], "k1");
  CHECK_EQ(input_shapes_.size(), 2U) << "matmul's input shape should be 2.\n";
  CHECK_EQ(input_shapes_[0].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[1].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[0][1], input_shapes_[1][0]) << "matmul's reduce axis shape should be same\n";
  auto C = Compute(
      {Expr(input_shapes_[0][0]), Expr(input_shapes_[1][1])},
      [&](Var i, Var j) { return ReduceSum(inputs[0](i, k1) * inputs[1](k1, j), {k1}); },
      "C");
  (*stages)->InsertLazily(C);
  int bn                                    = 32;
  auto [i_outer, i_inner, j_outer, j_inner] = (*stages)[C]->Tile(0, 1, bn, bn);  // NOLINT
  auto [k_outer, k_inner]                   = (*stages)[C]->Split(k1->name, 4);  // NOLINT
  (*stages)[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
  (*stages)[C]->Vectorize(j_inner, 8);

  outs.push_back(C);
  return outs;
}

// loop permutation
std::vector<ir::Tensor> MatmulLoopPermutationTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                            poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  auto k1 = Var(input_shapes_[0][1], "k1");
  CHECK_EQ(input_shapes_.size(), 2U) << "matmul's input shape should be 2.\n";
  CHECK_EQ(input_shapes_[0].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[1].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[0][1], input_shapes_[1][0]) << "matmul's reduce axis shape should be same\n";
  auto C = Compute(
      {Expr(input_shapes_[0][0]), Expr(input_shapes_[1][1])},
      [&](Var i, Var j) { return ReduceSum(inputs[0](i, k1) * inputs[1](k1, j), {k1}); },
      "C");
  (*stages)->InsertLazily(C);
  int bn                                    = 32;
  auto [i_outer, i_inner, j_outer, j_inner] = (*stages)[C]->Tile(0, 1, bn, bn);  // NOLINT
  auto [k_outer, k_inner]                   = (*stages)[C]->Split(k1->name, 4);  // NOLINT
  (*stages)[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
  (*stages)[C]->Vectorize(j_inner, 8);
  (*stages)[C]->Unroll(5);

  outs.push_back(C);
  return outs;
}

// array packing
std::vector<ir::Tensor> MatmulArrayPackingTester::CreateSpecificStrategy(const std::vector<ir::Tensor> &inputs,
                                                                         poly::StageMap *stages) {
  CHECK_EQ(inputs.size(), 2U) << "matmul's input tensor should be 2.\n";
  std::vector<ir::Tensor> outs;
  CHECK_EQ(input_shapes_.size(), 2U) << "matmul's input shape should be 2.\n";
  CHECK_EQ(input_shapes_[0].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[1].size(), 2U) << "matmul's input teosor's shape should be 2.\n";
  CHECK_EQ(input_shapes_[0][1], input_shapes_[1][0]) << "matmul's reduce axis shape should be same\n";

  Var k(input_shapes_[0][1], "k0");

  Expr bn(32);

  auto C_init = Compute(
      {Expr(input_shapes_[0][0]), Expr(input_shapes_[1][1])}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto packedB = Compute(
      {Expr(input_shapes_[1][1]) / bn, Expr(input_shapes_[0][1]), bn},
      [&](Expr x, Expr y, Expr z) { return inputs[1](y, x * bn + z); },
      "packedB");
  auto C = Compute(
      {Expr(input_shapes_[0][0]), Expr(input_shapes_[1][1])},
      [&](Expr i, Expr j) { return ReduceSum(inputs[0](i, k) * packedB(j / bn, k, j % bn), {k}); },
      "C");
  (*stages)->InsertLazily(C);
  (*stages)->InsertLazily(C_init);
  (*stages)->InsertLazily(packedB);

  (*stages)[C]->ShareBufferWith((*stages)[C_init]);
  (*stages)[C]->CtrlDepend(C_init);
  (*stages)[packedB]->Vectorize(2, 8);

  {
    auto [i_outer, i_inner, j_outer, j_inner] = (*stages)[C]->Tile(0, 1, bn.as_int32(), bn.as_int32());  // NOLINT
    auto [k_outer, k_inner]                   = (*stages)[C]->Split("k0", 4);                            // NOLINT

    (*stages)[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    (*stages)[C]->Vectorize(j_inner, 8);
  }
  outs.push_back(packedB);
  outs.push_back(C);
  return outs;
}

TEST(test_matmul, default) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_default", attrs, type);
}

TEST(test_matmul, tile) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulTileTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_tile", attrs, type, false);
}

TEST(test_matmul, split) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulSplitTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_split", attrs, type, false);
}

TEST(test_matmul, block) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulBlockTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_block", attrs, type, false);
}

TEST(test_matmul, vectorize) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulVectorizeTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_vectorize", attrs, type, false);
}

TEST(test_matmul, loop_permutation) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulLoopPermutationTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_loop_permutation", attrs, type, false);
}

TEST(test_matmul, array_packing) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::vector<std::vector<int>> input_shapes{{M, K}, {K, N}};
  std::string op_name = "matmul";
  hlir::framework::NodeAttr attrs;
  MatmulArrayPackingTester matmul_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  matmul_tester.TestOp("matmul_array_packing", attrs, type, false);
}

}  // namespace tests
}  // namespace cinn

#include "cinn/hlir/instruction/instruction.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace instruction {

TEST(Instruction, basic0) {
  auto A = CreateInput({1, 64, 112, 112}, "A");
  auto B = CreateInput({1, 64, 112, 112}, "B");
  auto C = CreateInput({1, 64, 112, 112}, "C");
  auto D = CreateInput({1, 64, 112, 112}, "D");

  auto E = Add(A, B);
  auto F = Add(C, D);
  auto G = Add(E, F);
  auto H = Relu(G);
  LOG(INFO) << "basic0 func: " << H->GetFunction();
}

TEST(Instruction, basic1) {
  auto input0  = CreateInput({1, 3, 224, 224}, "A");
  auto weights = CreateInput({64, 3, 7, 7}, "W");
  auto B       = CreateInput({1, 64, 112, 112}, "B");

  auto C = Pad(input0, {0, 0, 3, 3});
  auto D = ConvBroadcast(C, weights, {2, 2, 1, 1});
  auto E = ReduceSum(D, {4, 5, 6});
  auto F = Add(E, B);
  LOG(INFO) << "basic1 func: " << F->GetFunction();
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
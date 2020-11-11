#include "tests/benchmark/test_elementwise.h"

#include "cinn/cinn.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace tests {

TEST(test_elementwise_add, default_fp32) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  auto input_tensors = add_tester.CreateInputTensors<float>();
  add_tester.TestOp("elementwise_add_default_fp32", &input_tensors, attrs, type);
}

TEST(test_elementwise_add, default_int32) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> out_types{Int(32)};
  auto input_tensors = add_tester.CreateInputTensors<int>();
  add_tester.TestOp("elementwise_add_default_int32", &input_tensors, attrs, out_types);
  add_tester.Compare<int>();
}

}  // namespace tests
}  // namespace cinn

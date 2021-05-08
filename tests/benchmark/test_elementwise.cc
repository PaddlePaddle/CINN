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
  std::vector<Type> input_types{Float(32), Float(32)};
  std::vector<Type> output_types{Float(32)};
  auto input_tensors = add_tester.CreateInputTensors<float>();
  add_tester.TestOp("elementwise_add_default_fp32", input_tensors, attrs, input_types, output_types);
}

TEST(test_elementwise_add, default_int32) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> input_types{Int(32), Int(32)};
  std::vector<Type> output_types{Int(32)};
  auto input_tensors = add_tester.CreateInputTensors<int>();
  add_tester.TestOp("elementwise_add_default_int32", input_tensors, attrs, input_types, output_types);
  add_tester.Compare<int>();
}

}  // namespace tests
}  // namespace cinn

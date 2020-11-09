#include "tests/benchmark/test_elementwise.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace tests {

void ElementwiseAddTester::Compare() {
  std::vector<float *> all_datas = GetAllDatas();
  int out_dims                   = GetOutDims();
  CHECK_EQ(all_datas.size(), 3U) << "elementwise_add should have 3 args.\n";
  for (int i = 0; i < out_dims; ++i) {
    EXPECT_EQ((all_datas[0][i] + all_datas[1][i]), all_datas[2][i]);
  }
}

TEST(test_elementwise_add, default) {
  int M = 100;
  int N = 32;
  std::vector<std::vector<int>> input_shapes{{M, N}, {M, N}};
  std::string op_name = "elementwise_add";
  hlir::framework::NodeAttr attrs;
  ElementwiseAddTester add_tester(op_name, input_shapes);
  std::vector<Type> type{Float(32)};
  add_tester.TestOp("elementwise_add_default", attrs, type);
}

}  // namespace tests
}  // namespace cinn

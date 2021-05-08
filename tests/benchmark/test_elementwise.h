#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tests/benchmark/test_utils.h"

namespace cinn {
namespace tests {

class ElementwiseAddTester : public OpBenchmarkTester {
 public:
  ElementwiseAddTester(const std::string &op_name,
                       const std::vector<std::vector<int>> &input_shapes,
                       const common::Target &target = common::DefaultHostTarget(),
                       int repeat                   = 10,
                       float diff                   = 1e-5)
      : OpBenchmarkTester(op_name, input_shapes, target, repeat, diff) {}

  template <typename T>
  void Compare() {
    auto all_args = GetAllArgs();
    std::vector<T *> all_datas;
    for (auto &arg : all_args) {
      auto *buffer = cinn_pod_value_to_buffer_p(&arg);
      all_datas.push_back(reinterpret_cast<T *>(buffer->memory));
    }

    int out_dims = GetOutDims();
    CHECK_EQ(all_datas.size(), 3U) << "elementwise_add should have 3 args.\n";
    for (int i = 0; i < out_dims; ++i) {
      EXPECT_EQ(all_datas[0][i] + all_datas[1][i], all_datas[2][i]);
    }
  }
};

}  // namespace tests
}  // namespace cinn

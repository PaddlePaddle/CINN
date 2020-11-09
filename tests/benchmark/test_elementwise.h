#pragma once

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

  void Compare() override;
};

}  // namespace tests
}  // namespace cinn

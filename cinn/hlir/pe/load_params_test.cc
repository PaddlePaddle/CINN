#include <gtest/gtest.h>

#include "cinn/hlir/pe/schedule.h"

namespace cinn {
namespace hlir {
namespace pe {
using ir::Tensor;

TEST(load_x86_params, load_x86_params) {
  std::unordered_map<std::string, int> conv2d_factors;
  auto target                    = common::DefaultHostTarget();
  std::vector<int> shape_input   = {1, 64, 56, 56};
  std::vector<int> shape_weights = {64, 64, 3, 3};
  std::vector<int> strides       = {1, 1};
  std::vector<int> pads          = {1, 1};
  std::vector<int> dilations     = {1, 1};
  auto key                       = GenerateX86ConvKey(shape_input, shape_weights, strides, pads, dilations);
  GetConv2dFactors(&conv2d_factors, -1, -1, -1, -1, -1, Float(32), target, key);
  int ic_bn_size = conv2d_factors["ic_bn"];
  int oc_bn_size = conv2d_factors["oc_bn"];
  int fc_bn_size = conv2d_factors["fc_bn"];
  int ow_bn_size = conv2d_factors["ow_bn"];
  int unroll_kw  = conv2d_factors["unroll_kw"];
  ASSERT_EQ(ic_bn_size, 64);
  ASSERT_EQ(fc_bn_size, 64);
  ASSERT_EQ(oc_bn_size, 32);
  ASSERT_EQ(ow_bn_size, 7);
  ASSERT_EQ(unroll_kw, 1);
}

TEST(load_cuda_params, load_cuda_params) {
  auto &res = ScheduleParam::get_cuda_instance().GetParam();
  if (res.empty()) {
    CreateCudaSerialData();
    LoadSerialData(&res);
  }
  std::string key = "CudaScheduleConv 1 3 230 230 64 3 7 7 1 64 112 112";
  ASSERT_EQ(res.count(key), 1);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "tests/benchmark/test_utils.h"

namespace cinn {
namespace tests {
using AttrType = std::variant<bool,
                              float,
                              int,
                              std::string,
                              std::vector<bool>,
                              std::vector<int>,
                              std::vector<float>,
                              std::vector<std::string>>;

#define TEST_DEFAULT(op_name__, shape_name__, type__)                          \
  TEST(op_defualt, shape_name__) {                                             \
    std::vector<std::vector<int>> input_shapes = shapes_##shape_name__;        \
    std::string op_name                        = #op_name__;                   \
    hlir::framework::NodeAttr attrs;                                           \
    OpBenchmarkTester tester(op_name, input_shapes);                           \
    auto input_tensors = tester.CreateInputTensors<float>();                   \
    tester.TestOp(common::UniqName(#op_name__), input_tensors, attrs, type__); \
  }

#define TEST_DEFAULT1(op_name__, shape_name__, type__, attr_store__)           \
  TEST(op_defualt1, shape_name__) {                                            \
    std::vector<std::vector<int>> input_shapes = shapes_##shape_name__;        \
    std::string op_name                        = #op_name__;                   \
    OpBenchmarkTester tester(op_name, input_shapes);                           \
    hlir::framework::NodeAttr attrs;                                           \
    attrs.attr_store   = attr_store__;                                         \
    auto input_tensors = tester.CreateInputTensors<float>();                   \
    tester.TestOp(common::UniqName(#op_name__), input_tensors, attrs, type__); \
  }

std::vector<Type> type{Float(32)};
std::vector<Type> type1{Float(32), Float(32)};
// add
// std::vector<std::vector<int>> shapes_add = {{1024, 1024, 1024}, {1024, 1024, 1024}};
// TEST_DEFAULT(elementwise_add, add, type)
std::vector<std::vector<int>> shapes_add1 = {{100, 32}, {100, 32}};
TEST_DEFAULT(elementwise_add, add1, type)
std::vector<std::vector<int>> shapes_add2 = {{1024, 14, 14}, {1024, 14, 14}};
TEST_DEFAULT(elementwise_add, add2, type)
std::vector<std::vector<int>> shapes_add3 = {{1}, {1}};
TEST_DEFAULT(elementwise_add, add3, type)
// mul
// std::vector<std::vector<int>> shapes_mul = {{1024, 1024, 1024}, {1024, 1024, 1024}};
// TEST_DEFAULT(elementwise_mul, mul, type)
std::vector<std::vector<int>> shapes_mul1 = {{100, 32}, {100, 32}};
TEST_DEFAULT(elementwise_mul, mul1, type)
std::vector<std::vector<int>> shapes_mul2 = {{1024, 14, 14}, {1024, 14, 14}};
TEST_DEFAULT(elementwise_mul, mul2, type)
std::vector<std::vector<int>> shapes_mul3 = {{1}, {1}};
TEST_DEFAULT(elementwise_mul, mul3, type)

// relu
std::vector<std::vector<int>> shapes_relu = {{2, 512, 7, 7}};
TEST_DEFAULT(relu, relu, type)
std::vector<std::vector<int>> shapes_relu1 = {{1024, 14, 14}};
TEST_DEFAULT(relu, relu1, type)

// conv2d nchw
std::vector<std::vector<int>> shapes_conv2d_nchw = {{2, 512, 7, 7}, {512, 512, 3, 3}};
std::vector<int> padding_conv2d({0, 0});
std::vector<int> stride_conv2d({1, 1});
std::vector<int> dilation_conv2d({1, 1});
std::unordered_map<std::string, AttrType> attr_store_conv2d = {
    {"padding", padding_conv2d}, {"stride", stride_conv2d}, {"dilation", dilation_conv2d}};
TEST_DEFAULT1(conv2d, conv2d_nchw, type, attr_store_conv2d)
std::vector<std::vector<int>> shapes_conv2d_nchw1 = {{2, 1024, 14, 14}, {256, 1024, 1, 1}};
TEST_DEFAULT1(conv2d, conv2d_nchw1, type, attr_store_conv2d)

// depthwise_conv2d nchw
std::vector<std::vector<int>> shapes_depthwise_conv2d_nchw            = {{2, 32, 112, 112}, {32, 1, 3, 3}};
std::vector<int> stride_depthwise_conv2d                              = {1, 1};
std::vector<int> padding_depthwise_conv2d                             = {1, 1};
std::vector<int> dilation_depthwise_conv2d                            = {1, 1};
std::unordered_map<std::string, AttrType> attr_store_depthwise_conv2d = {{"padding", padding_depthwise_conv2d},
                                                                         {"stride", stride_depthwise_conv2d},
                                                                         {"dilation", dilation_depthwise_conv2d}};
TEST_DEFAULT1(depthwise_conv2d, depthwise_conv2d_nchw, type, attr_store_depthwise_conv2d)

// pool2d
hlir::framework::NodeAttr attrs;
std::vector<int> kernel_size                                = {3, 3};
std::vector<int> stride_size                                = {2, 2};
std::vector<int> padding_size                               = {1, 1, 1, 1};
std::string pool_type                                       = "max";
std::unordered_map<std::string, AttrType> attr_store_pool2d = {{"kernel_size", kernel_size},
                                                               {"stride_size", stride_size},
                                                               {"padding_size", padding_size},
                                                               {"pool_type", pool_type}};

std::vector<std::vector<int>> shapes_pool2d = {{2, 64, 112, 112}};
TEST_DEFAULT1(pool2d, pool2d, type, attr_store_pool2d)
std::vector<std::vector<int>> shapes_pool2d1 = {{2, 1024, 14, 14}};
TEST_DEFAULT1(pool2d, pool2d1, type, attr_store_pool2d)

// softmax
// std::vector<std::vector<int>> shapes_softmax = {{1024,2048}};
// TEST_DEFAULT(softmax, softmax, type1)
std::vector<std::vector<int>> shapes_softmax1 = {{3, 1000}};
TEST_DEFAULT(softmax, softmax1, type1)

// sigmoid
std::vector<std::vector<int>> shapes_sigmoid = {{2, 672, 1, 1}};
TEST_DEFAULT(sigmoid, sigmoid, type)
std::vector<std::vector<int>> shapes_sigmoid1 = {{3, 1000}};
TEST_DEFAULT(sigmoid, sigmoid1, type)

// matmul
std::vector<std::vector<int>> shapes_matmul = {{32, 32}, {32, 32}};
TEST_DEFAULT(matmul, matmul, type)
// std::vector<std::vector<int>> shapes_matmul1 = {{512,512}, {512,512}};
// TEST_DEFAULT(matmul, matmul1, type)
// std::vector<std::vector<int>> shapes_matmul2 = {{100,32}, {32,100}};
// TEST_DEFAULT(matmul, matmul2, type)
// std::vector<std::vector<int>> shapes_matmul3 = {{1024,1024}, {1024,1024}};
// TEST_DEFAULT(matmul, matmul3, type)

// batchnorm
std::vector<std::vector<int>> shapes_batchnorm = {{2, 32, 112, 112}, {32}, {32}, {32}, {32}};
TEST_DEFAULT(batchnorm, batchnorm, type)

// scale
std::vector<std::vector<int>> shapes_scale = {{2, 1000}};
TEST_DEFAULT(scale, scale, type)

// slice
std::vector<std::vector<int>> shapes_slice = {{2, 32, 113, 113}};
std::vector<int> starts({1, 1});
std::vector<int> ends({10000000, 10000000});
std::vector<int> axes({2, 3});
std::unordered_map<std::string, AttrType> attr_store_slice = {{"starts", starts}, {"ends", ends}, {"axes", axes}};
TEST_DEFAULT1(slice, slice, type, attr_store_slice)

}  // namespace tests
}  // namespace cinn

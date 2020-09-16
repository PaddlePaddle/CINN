#include "cinn/hlir/pe/nn.h"

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/elementwise.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForRelu(const framework::NodeAttr &attrs,
                                            const std::vector<ir::Tensor> &inputs,
                                            const std::vector<Type> &out_type,
                                            const Target &target) {
  framework::CINNCompute relu_compute([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of relu compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for relu compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Relu<float>(A.as_tensor_ref(), 0.0, UniqName("Relu_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule relu_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of relu schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of relu op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(relu_compute, relu_schedule, "strategy.relu.x86", 1);
  } else {
    LOG(FATAL) << "Relu op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<framework::shape_t> InferShapeForRelu(const std::vector<framework::shape_t> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForRelu(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForRelu6(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const Target &target) {
  framework::CINNCompute relu_compute([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of relu6 compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for relu6 compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Relu6<float>(A.as_tensor_ref(), 0.0, UniqName("Relu6_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule relu_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of relu6 schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of relu6 op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(relu_compute, relu_schedule, "strategy.relu6.x86", 1);
  } else {
    LOG(FATAL) << "Relu6 op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForConv2d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  std::vector<int> padding({0, 0});
  std::vector<int> stride({1, 1});
  std::vector<int> dilation({1, 1});
  int groups(1);
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = std::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = std::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = std::get<std::vector<int>>(attrs.attr_store.at("dilation"));
  }
  if (attrs.attr_store.find("groups") != attrs.attr_store.end()) {
    groups = std::get<int>(attrs.attr_store.at("groups"));
  }
  framework::CINNCompute conv2d_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of conv2d compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for conv2d compute\n";
    Expr A = a[0];
    Expr B = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK_EQ(padding.size(), 2) << "The size of padding in conv2d op is not 2! Please check.";
    CHECK_EQ(stride.size(), 2) << "The size of stride in conv2d op is not 2! Please check.";
    auto out    = pe::Conv2d_NCHW(A.as_tensor_ref(),
                               B.as_tensor_ref(),
                               padding[0],
                               padding[1],
                               stride[0],
                               stride[1],
                               dilation[0],
                               dilation[1],
                               groups,
                               UniqName("Conv2d_output"));
    auto stages = CreateStages(out);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule conv2d_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of conv2d schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 4UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of conv2d op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(conv2d_compute, conv2d_schedule, "strategy.conv2d.x86", 1);
  } else {
    LOG(FATAL) << "Conv2d op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<shape_t> InferShapeForConv2d(const std::vector<shape_t> &inputs_shape, const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<int> padding({0, 0});
  std::vector<int> stride({1, 1});
  std::vector<int> dilation({1, 1});
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = std::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = std::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = std::get<std::vector<int>>(attrs.attr_store.at("dilation"));
  }
  CHECK_EQ(padding.size(), 2) << "The size of padding in conv2d op is not 2! Please check.";
  CHECK_EQ(stride.size(), 2) << "The size of stride in conv2d op is not 2! Please check.";
  CHECK_GE(inputs_shape[0].size(), 3) << "The first input tensor's shape size of conv2d op is < 3! Please check.";

  int out_shape_h =
      (inputs_shape[0][2] - ((inputs_shape[1][2] - 1) * dilation[0] + 1) + 2 * padding[0]) / stride[0] + 1;
  int out_shape_w =
      (inputs_shape[0][3] - ((inputs_shape[1][3] - 1) * dilation[1] + 1) + 2 * padding[1]) / stride[1] + 1;
  std::vector<shape_t> res{{inputs_shape[0][0],
                            inputs_shape[0][1],
                            inputs_shape[0][2] + 2 * padding[0],
                            inputs_shape[0][3] + 2 * padding[1]},
                           {inputs_shape[1][0],
                            inputs_shape[1][1],
                            (inputs_shape[1][2] - 1) * dilation[0] + 1,
                            (inputs_shape[1][3] - 1) * dilation[1] + 1},
                           {inputs_shape[0][0], inputs_shape[1][0], out_shape_h, out_shape_w}};
  return res;
}

std::vector<Type> InferDtypeForConv2d(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[1], inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForBatchNorm(const framework::NodeAttr &attrs,
                                                 const std::vector<ir::Tensor> &inputs,
                                                 const std::vector<Type> &out_type,
                                                 const Target &target) {
  float epsilon = 0.00001f;
  if (attrs.attr_store.find("epsilon") != attrs.attr_store.end()) {
    epsilon = std::get<float>(attrs.attr_store.at("epsilon"));
  }
  framework::CINNCompute batchnorm_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of batchnorm compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 5U) << "at least 5 input tensors for batchnorm compute\n";
    Expr A        = a[0];
    Expr Scale    = a[1];
    Expr Bias     = a[2];
    Expr Mean     = a[3];
    Expr Variance = a[4];

    CHECK(A.as_tensor());
    CHECK(Scale.as_tensor());
    CHECK(Bias.as_tensor());
    CHECK(Mean.as_tensor());
    CHECK(Variance.as_tensor());
    auto out    = pe::BatchNorm_NCHW(A.as_tensor_ref(),
                                  Scale.as_tensor_ref(),
                                  Bias.as_tensor_ref(),
                                  Mean.as_tensor_ref(),
                                  Variance.as_tensor_ref(),
                                  epsilon,
                                  UniqName("BatchNorm_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule batchnorm_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of batchnorm schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of batchnorm op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(batchnorm_compute, batchnorm_schedule, "strategy.batchnorm.x86", 1);
  } else {
    LOG(FATAL) << "BatchNorm op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<shape_t> InferShapeForBatchNorm(const std::vector<shape_t> &inputs_shape,
                                            const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForBatchNorm(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForPool1d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  framework::CINNCompute pool1d_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool1d compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensor of pool1d compute is empty! Please check.\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto attr_store = attrs.attr_store;
    std::vector<int> kernel_size;   // [kernel_w]
    std::vector<int> stride_size;   // [stride_w]
    std::vector<int> padding_size;  // [padding_left, padding_right]
    std::string pool_type   = "max";
    bool ceil_mode          = false;
    bool exclusive          = true;
    std::string data_format = "NCW";
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "kernel_size") {
        kernel_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "stride_size") {
        stride_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "padding_size") {
        padding_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "pool_type") {
        pool_type = std::get<std::string>(iter.second);
      } else if (iter.first == "ceil_mode") {
        ceil_mode = std::get<bool>(iter.second);
      } else if (iter.first == "exclusive") {
        exclusive = std::get<bool>(iter.second);
      } else if (iter.first == "data_format") {
        data_format = std::get<std::string>(iter.second);
      } else {
        LOG(ERROR) << "unsupported attr: " << iter.first << std::endl;
      }
    }
    CHECK(!kernel_size.empty()) << "kernel_size for pool1d is empty. Please check.\n";
    CHECK(!stride_size.empty()) << "stride_size for pool1d is empty. Please check.\n";
    CHECK(!padding_size.empty()) << "padding_size for pool1d is empty. Please check.\n";

    auto out = pe::Pool1d(A.as_tensor_ref(),
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          UniqName("T_Pool1d_out"));

    auto stages = CreateStages(out);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool1d_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool1d schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool1d_compute, pool1d_schedule, "strategy.pool1d.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForPool1d(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  auto attr_store = attrs.attr_store;
  std::vector<int> kernel_size;   // [kernel_w]
  std::vector<int> stride_size;   // [stride_w]
  std::vector<int> padding_size;  // [padding_left, padding_right]
  std::string pool_type   = "max";
  bool ceil_mode          = false;
  bool exclusive          = true;
  std::string data_format = "NCW";
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "kernel_size") {
      kernel_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = std::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = std::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = std::get<std::string>(iter.second);
    }
  }
  CHECK_EQ(kernel_size.size(), 1U) << "kernel size for pool1d should be 1.\n";
  CHECK_EQ(stride_size.size(), 1U) << "stride_size size for pool1d should be 1.\n";
  CHECK_EQ(padding_size.size(), 2U) << "padding_size size for pool1d should be 2.\n";

  std::vector<int> output_shape0 = inputs_shape[0];
  std::vector<int> output_shape1 = inputs_shape[0];
  CHECK_EQ(output_shape0.size(), 3U);
  int width_axis = -1;
  if (data_format == "NCW") {
    width_axis = 2;
  } else if (data_format == "NWC") {
    width_axis = 1;
  } else {
    LOG(ERROR) << "unsupported data_format: " << data_format << std::endl;
  }

  output_shape0[width_axis] += padding_size[0] + padding_size[1];
  if (ceil_mode) {
    output_shape0[width_axis] += stride_size[0];
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[0] + padding_size[0] + padding_size[1] + stride_size[0] - 1) /
            stride_size[0] +
        1;
  } else {
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[0] + padding_size[0] + padding_size[1]) / stride_size[0] + 1;
  }

  std::vector<std::vector<int>> res{output_shape0, output_shape1};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForPool2d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  framework::CINNCompute pool2d_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool2d compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensor of pool2d compute is empty! Please check.\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto attr_store = attrs.attr_store;
    std::vector<int> kernel_size;   // [kernel_h, kernel_w]
    std::vector<int> stride_size;   // [stride_h, stride_w]
    std::vector<int> padding_size;  // [padding_top, padding_left, padding_bottom, padding_right]
    std::string pool_type   = "max";
    bool ceil_mode          = false;
    bool exclusive          = true;
    std::string data_format = "NCHW";
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "kernel_size") {
        kernel_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "stride_size") {
        stride_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "padding_size") {
        padding_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "pool_type") {
        pool_type = std::get<std::string>(iter.second);
      } else if (iter.first == "ceil_mode") {
        ceil_mode = std::get<bool>(iter.second);
      } else if (iter.first == "exclusive") {
        exclusive = std::get<bool>(iter.second);
      } else if (iter.first == "data_format") {
        data_format = std::get<std::string>(iter.second);
      } else {
        LOG(ERROR) << "unsupported attr: " << iter.first << std::endl;
      }
    }
    CHECK(!kernel_size.empty()) << "kernel_size for pool2d is empty. Please check.\n";
    CHECK(!stride_size.empty()) << "stride_size for pool2d is empty. Please check.\n";
    CHECK(!padding_size.empty()) << "padding_size for pool2d is empty. Please check.\n";

    auto out = pe::Pool2d(A.as_tensor_ref(),
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          UniqName("T_Pool2d_out"));

    auto stages = CreateStages(out);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool2d_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool2d schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool2d_compute, pool2d_schedule, "strategy.pool2d.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForPool2d(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  auto attr_store = attrs.attr_store;
  std::vector<int> kernel_size;
  std::vector<int> stride_size;
  std::vector<int> padding_size;
  std::string pool_type   = "max";
  bool ceil_mode          = false;
  bool exclusive          = true;
  std::string data_format = "NCHW";
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "kernel_size") {
      kernel_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = std::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = std::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = std::get<std::string>(iter.second);
    }
  }
  CHECK_EQ(kernel_size.size(), 2U) << "kernel size for pool1d should be 2.\n";
  CHECK_EQ(stride_size.size(), 2U) << "stride_size size for pool1d should be 2.\n";

  std::vector<int> output_shape0 = inputs_shape[0];
  std::vector<int> output_shape1 = inputs_shape[0];
  CHECK_EQ(inputs_shape[0].size(), 4U) << "input_shape size for pool2d should be 4.\n";
  int height_axis = -1;
  int width_axis  = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis  = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis  = 2;
  } else {
    LOG(ERROR) << "unsupported data_format: " << data_format << std::endl;
  }

  output_shape0[height_axis] += padding_size[0] + padding_size[2];
  output_shape0[width_axis] += padding_size[1] + padding_size[3];
  if (ceil_mode) {
    output_shape0[height_axis] += stride_size[0] - 1;
    output_shape0[width_axis] += stride_size[1] - 1;
    output_shape1[height_axis] =
        (inputs_shape[0][height_axis] - kernel_size[0] + padding_size[0] + padding_size[2] + stride_size[0] - 1) /
            stride_size[0] +
        1;
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[1] + padding_size[1] + padding_size[3] + stride_size[1] - 1) /
            stride_size[1] +
        1;
  } else {
    output_shape1[height_axis] =
        (inputs_shape[0][height_axis] - kernel_size[0] + padding_size[0] + padding_size[2]) / stride_size[0] + 1;
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[1] + padding_size[1] + padding_size[3]) / stride_size[1] + 1;
  }

  std::vector<std::vector<int>> res{output_shape0, output_shape1};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForPool3d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  framework::CINNCompute pool3d_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool3d compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensor of pool3d compute is empty! Please check.\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto attr_store = attrs.attr_store;
    std::vector<int> kernel_size;  // [kernel_d, kernel_h, kernel_w]
    std::vector<int> stride_size;  // [stride_d, stride_h, stride_w]
    std::vector<int>
        padding_size;  // [padding_front, padding_top, padding_left, padding_back, padding_bottom, padding_right]
    std::string pool_type   = "max";
    bool ceil_mode          = false;
    bool exclusive          = true;
    std::string data_format = "NCDHW";
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "kernel_size") {
        kernel_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "stride_size") {
        stride_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "padding_size") {
        padding_size = std::get<std::vector<int>>(iter.second);
      } else if (iter.first == "pool_type") {
        pool_type = std::get<std::string>(iter.second);
      } else if (iter.first == "ceil_mode") {
        ceil_mode = std::get<bool>(iter.second);
      } else if (iter.first == "exclusive") {
        exclusive = std::get<bool>(iter.second);
      } else if (iter.first == "data_format") {
        data_format = std::get<std::string>(iter.second);
      } else {
        LOG(ERROR) << "unsupported attr: " << iter.first << std::endl;
      }
    }
    CHECK(!kernel_size.empty()) << "kernel_size for pool3d is empty. Please check.\n";
    CHECK(!stride_size.empty()) << "stride_size for pool3d is empty. Please check.\n";
    CHECK(!padding_size.empty()) << "padding_size for pool3d is empty. Please check.\n";

    auto out = pe::Pool3d(A.as_tensor_ref(),
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          UniqName("T_Pool3d_out"));

    auto stages = CreateStages(out);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool3d_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool3d schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool3d_compute, pool3d_schedule, "strategy.pool3d.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForPool3d(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  auto attr_store = attrs.attr_store;
  std::vector<int> kernel_size;  // [kernel_d, kernel_h, kernel_w]
  std::vector<int> stride_size;  // [stride_d, stride_h, stride_w]
  std::vector<int>
      padding_size;  // [padding_front, padding_top, padding_left, padding_bottom, padding_right, padding_back]
  std::string pool_type   = "max";
  bool ceil_mode          = false;
  bool exclusive          = true;
  std::string data_format = "NCDHW";
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "kernel_size") {
      kernel_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = std::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = std::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = std::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = std::get<std::string>(iter.second);
    }
  }

  CHECK_EQ(kernel_size.size(), 3U) << "kernel_size for pool3d should be 3.\n";
  CHECK_EQ(stride_size.size(), 3U) << "stride_size for pool3d should be 3.\n";

  std::vector<int> output_shape0 = inputs_shape[0];
  std::vector<int> output_shape1 = inputs_shape[0];
  CHECK_EQ(inputs_shape[0].size(), 6U) << "input_shape size for pool3d should be 6.\n";
  int depth_axis  = -1;
  int height_axis = -1;
  int width_axis  = -1;
  if (data_format == "NCDHW") {
    depth_axis  = 2;
    height_axis = 3;
    width_axis  = 4;
  } else if (data_format == "NDHWC") {
    depth_axis  = 1;
    height_axis = 2;
    width_axis  = 3;
  } else {
    LOG(ERROR) << "unsupported data_format: " << data_format << std::endl;
  }

  output_shape0[depth_axis] += padding_size[0] + padding_size[3];
  output_shape0[height_axis] += padding_size[1] + padding_size[4];
  output_shape0[width_axis] += padding_size[2] + padding_size[5];
  if (ceil_mode) {
    output_shape0[depth_axis] += stride_size[0] - 1;
    output_shape0[height_axis] += stride_size[1] - 1;
    output_shape0[width_axis] += stride_size[2] - 1;
    output_shape1[depth_axis] =
        (inputs_shape[0][depth_axis] - kernel_size[0] + padding_size[0] + padding_size[3] + stride_size[0] - 1) /
            stride_size[0] +
        1;
    output_shape1[height_axis] =
        (inputs_shape[0][height_axis] - kernel_size[1] + padding_size[1] + padding_size[4] + stride_size[1] - 1) /
            stride_size[1] +
        1;
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[2] + padding_size[2] + padding_size[5] + stride_size[2] - 1) /
            stride_size[2] +
        1;
  } else {
    output_shape1[depth_axis] =
        (inputs_shape[0][depth_axis] - kernel_size[0] + padding_size[0] + padding_size[3]) / stride_size[0] + 1;
    output_shape1[height_axis] =
        (inputs_shape[0][height_axis] - kernel_size[1] + padding_size[1] + padding_size[4]) / stride_size[1] + 1;
    output_shape1[width_axis] =
        (inputs_shape[0][width_axis] - kernel_size[2] + padding_size[2] + padding_size[5]) / stride_size[2] + 1;
  }

  std::vector<std::vector<int>> res{output_shape0, output_shape1};
  return res;
}

std::vector<Type> InferDtypeForPool(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForSigmoid(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const Target &target) {
  framework::CINNCompute sigmoid_compute([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of sigmoid compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for sigmoid compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Sigmoid(A.as_tensor_ref(), UniqName("Sigmoid_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule sigmoid_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of sigmoid schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of sigmoid op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(sigmoid_compute, sigmoid_schedule, "strategy.sigmoid.x86", 1);
  } else {
    LOG(FATAL) << "Sigmoid op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<framework::shape_t> InferShapeForSigmoid(const std::vector<framework::shape_t> &inputs_shape,
                                                     const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForSigmoid(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForSoftmax(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const Target &target) {
  int axis = -1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "axis") {
      axis = std::get<int>(iter.second);
    }
  }
  framework::CINNCompute softmax_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of softmax compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensors of softmax compute is empty! Please check.";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    int new_axis = axis;
    if (axis == -1) {
      new_axis = A->shape.size() - 1;
    }
    auto out = pe::Softmax(A, new_axis, UniqName("Softmax_output"));
    CHECK_EQ(out.size(), 2);
    auto stages = CreateStages(out);
    out[0]->InitReduction(stages, Expr(0.f));
    *ret = CINNValuePack{{CINNValue(out[0]), CINNValue(out[1]), CINNValue(stages)}};
  });

  framework::CINNSchedule softmax_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of softmax schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL) << "The input tensor's size of softmax schedule is " << arg_pack.size()
                                   << "and it should be equal to 3! Please check.";
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(softmax_compute, softmax_schedule, "strategy.softmax.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForSoftmax(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0], inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForSoftmax(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(nn_ops) {
  CINN_REGISTER_OP(relu)
      .describe("Output 0 for each input element < 0. Output itself for each input element >= 0.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForRelu)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForRelu))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForRelu))
      .set_support_level(4);

  CINN_REGISTER_OP(relu6)
      .describe("Output 0 for each input element < 0. Output itself for each input element >= 0 and <=6.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForRelu6)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForRelu))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForRelu))
      .set_support_level(4);

  CINN_REGISTER_OP(conv2d)
      .describe("Do a 2-D convolution with an NCHW-layout.")
      .set_num_inputs(2)  // here we consider filter as anohter input
      .set_num_outputs(3)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForConv2d)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForConv2d))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForConv2d))
      .set_support_level(4);

  CINN_REGISTER_OP(batchnorm)
      .describe("Can be used as a normalizer function for convolution or fully_connected operations.")
      .set_num_inputs(5)  // here we consider batchnorm's 4 attrs(mean, variance, scale, bias) as other 4 inputs
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBatchNorm)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForBatchNorm))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForBatchNorm))
      .set_support_level(4);

  CINN_REGISTER_OP(pool1d)
      .describe("Do pooling on the width dimension of the input tensor.")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForPool1d)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForPool1d))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForPool))
      .set_support_level(4);

  CINN_REGISTER_OP(pool2d)
      .describe("Do pooling on the height and width dimension of the input tensor.")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForPool2d)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForPool2d))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForPool))
      .set_support_level(4);

  CINN_REGISTER_OP(pool3d)
      .describe("Do pooling on the depth, height and width dimension of the input tensor.")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForPool3d)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForPool3d))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForPool))
      .set_support_level(4);

  CINN_REGISTER_OP(sigmoid)
      .describe("Apply sigmoid activation on input tensor. Y = 1 / (1 + Exp(-X))")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSigmoid)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForSigmoid))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForSigmoid))
      .set_support_level(4);

  CINN_REGISTER_OP(softmax)
      .describe("This operator implements the softmax layer")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSoftmax)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForSoftmax))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForSoftmax))
      .set_support_level(4);

  return true;
}

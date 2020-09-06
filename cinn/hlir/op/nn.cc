#include "cinn/hlir/pe/nn.h"

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForRelu(const framework::NodeAttr &attrs,
                                            const std::vector<ir::Tensor> &inputs,
                                            const std::vector<Type> &out_type,
                                            const Target &target) {
  framework::CINNCompute relu_compute([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    Expr A          = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Relu<float>(A.as_tensor_ref(), 0.0, UniqName("Relu_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule relu_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];  // NOLINT
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of relu op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(relu_compute, relu_schedule, "strategy.relu.x86", 1);
  } else {
    LOG(INFO) << "Relu op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<std::vector<int>> InferShapeForRelu(const std::vector<std::vector<int>> &inputs_shape,
                                                const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
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
    CINNValuePack a = args[0];
    Expr A          = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Relu6<float>(A.as_tensor_ref(), 0.0, UniqName("Relu6_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule relu_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];  // NOLINT
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of relu6 op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(relu_compute, relu_schedule, "strategy.relu6.x86", 1);
  } else {
    LOG(INFO) << "Relu6 op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForConv2d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  std::vector<int> padding({0, 0});
  std::vector<int> stride({1, 1});
  int dilation(1);
  int groups(1);
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = std::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = std::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = std::get<int>(attrs.attr_store.at("dilation"));
  }
  if (attrs.attr_store.find("groups") != attrs.attr_store.end()) {
    groups = std::get<int>(attrs.attr_store.at("groups"));
  }
  framework::CINNCompute conv2d_compute([=](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    Expr A          = a[0];
    Expr B          = a[1];
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
                               dilation,
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
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 4UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of conv2d op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(conv2d_compute, conv2d_schedule, "strategy.conv2d.x86", 1);
  } else {
    LOG(INFO) << "Conv2d op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<std::vector<int>> InferShapeForConv2d(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<int> padding({0, 0});
  std::vector<int> stride({1, 1});
  int dilation(1);
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = std::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = std::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = std::get<int>(attrs.attr_store.at("dilation"));
  }
  CHECK_EQ(padding.size(), 2) << "The size of padding in conv2d op is not 2! Please check.";
  CHECK_EQ(stride.size(), 2) << "The size of stride in conv2d op is not 2! Please check.";
  CHECK_EQ(inputs_shape[0].size(), 4) << "The first input tensor's shape size of conv2d op is not 4! Please check.";
  int out_shape_h = (inputs_shape[0][2] - ((inputs_shape[1][2] - 1) * dilation + 1) + 2 * padding[0]) / stride[0] + 1;
  int out_shape_w = (inputs_shape[0][3] - ((inputs_shape[1][3] - 1) * dilation + 1) + 2 * padding[1]) / stride[1] + 1;
  std::vector<std::vector<int>> res{{inputs_shape[0][0],
                                     inputs_shape[0][1],
                                     inputs_shape[0][2] + 2 * padding[0],
                                     inputs_shape[0][3] + 2 * padding[1]},
                                    {inputs_shape[1][0],
                                     inputs_shape[1][1],
                                     (inputs_shape[1][2] - 1) * dilation + 1,
                                     (inputs_shape[1][3] - 1) * dilation + 1},
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
    CINNValuePack a = args[0];
    Expr A          = a[0];
    Expr B          = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto out    = pe::BatchNorm_NCHW(A.as_tensor_ref(), B.as_tensor_ref(), epsilon, UniqName("BatchNorm_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule batchnorm_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of batchnorm op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(batchnorm_compute, batchnorm_schedule, "strategy.batchnorm.x86", 1);
  } else {
    LOG(INFO) << "BatchNorm op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<std::vector<int>> InferShapeForBatchNorm(const std::vector<std::vector<int>> &inputs_shape,
                                                     const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForBatchNorm(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
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
      .set_num_inputs(2)  // here we consider batchnorm's 4 attrs(mean, variance, scale, bias) as another input
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBatchNorm)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForBatchNorm))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForBatchNorm))
      .set_support_level(4);

  return true;
}

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

std::shared_ptr<OpStrategy> StrategyForAdd(const framework::NodeAttr &attrs,
                                           const std::vector<ir::Tensor> &inputs,
                                           const std::vector<Type> &out_type,
                                           const Target &target) {
  framework::CINNCompute add_compute([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    ir::Expr A      = a[0];
    ir::Expr B      = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto out = pe::Add(A.as_tensor_ref(), B.as_tensor_ref(), UniqName("C"));

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(ir::Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule add_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack      = args[0];
    ir::Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(add_compute, add_schedule, "strategy.add.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForAdd(const std::vector<std::vector<int>> &inputs_shape,
                                               const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForAdd(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForRelu(const framework::NodeAttr &attrs,
                                            const std::vector<ir::Tensor> &inputs,
                                            const std::vector<Type> &out_type,
                                            const Target &target) {
  for (auto iter : attrs.attr_store) {
    LOG(INFO) << "In StrategyForRelu, the input NodeAttr attrs is: ";
    LOG(INFO) << iter.first << " : " << std::get<int>(iter.second);
  }
  framework::CINNCompute relu_compute([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    ir::Expr A      = a[0];
    CHECK(A.as_tensor());
    auto out    = pe::Relu<float>(A.as_tensor_ref(), 0.0, UniqName("Relu_output"));
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(ir::Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule relu_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack      = args[0];
    ir::Expr A [[maybe_unused]] = arg_pack[0];
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

std::shared_ptr<OpStrategy> StrategyForConv2d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const Target &target) {
  framework::CINNCompute conv2d_compute([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    ir::Expr A      = a[0];
    ir::Expr B      = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    LOG(INFO) << "before pe::Conv2d_nchw";
    auto out = pe::Conv2d_nchw(A.as_tensor_ref(), B.as_tensor_ref(), 1, 1, 2, 2, 2, 1);
    LOG(INFO) << "after pe::Conv2d_nchw";
    auto stages = CreateStages(out);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(ir::Expr(t.get())));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule conv2d_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack      = args[0];
    ir::Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 4UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of conv2d op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(conv2d_compute, conv2d_schedule, "strategy.relu.x86", 1);
  } else {
    LOG(INFO) << "Conv2d op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<std::vector<int>> InferShapeForConv2d(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  auto dilation   = std::get<int>(attrs.attr_store.at("dilation"));
  auto padding    = std::get<std::vector<int>>(attrs.attr_store.at("padding"));
  auto stride     = std::get<std::vector<int>>(attrs.attr_store.at("stride"));
  int out_shape_h = (inputs_shape[0][2] - ((inputs_shape[1][2] - 1) * dilation + 1) + 2 * padding[0]) / stride[0] + 1;
  int out_shape_w = (inputs_shape[0][3] - ((inputs_shape[1][3] - 1) * dilation + 1) + 2 * padding[1]) / stride[1] + 1;
  std::vector<std::vector<int>> res{{inputs_shape[0][0], inputs_shape[1][0], out_shape_h, out_shape_w}};
  return res;
}

std::vector<Type> InferDtypeForConv2d(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(nn_ops) {
  CINN_REGISTER_OP(add)
      .describe("Add two tensors")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForAdd)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForAdd))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForAdd))
      .set_support_level(4);
  CINN_REGISTER_OP(relu)
      .describe("Output 0 for each input element < 0. Output itself for each input element >= 0.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForRelu)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForRelu))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForRelu))
      .set_support_level(4);
  CINN_REGISTER_OP(conv2d)
      .describe("Do a 2-D convolution with an NCHW-layout.")
      .set_num_inputs(2)  // here we consider filter as anohter input
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForConv2d)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForConv2d))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForConv2d))
      .set_support_level(4);
}

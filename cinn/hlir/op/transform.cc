#include "cinn/hlir/pe/transform.h"

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForMul(const framework::NodeAttr &attrs,
                                           const std::vector<ir::Tensor> &inputs,
                                           const std::vector<Type> &out_type,
                                           const Target &target) {
  framework::CINNCompute add_compute([&attrs](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    Expr A          = a[0];
    Expr B          = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto attr_store    = attrs.attr_store;
    bool trans_a       = false;
    bool trans_b       = false;
    int x_num_col_dims = 1;
    int y_num_col_dims = 1;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "trans_a") {
        trans_a = std::get<bool>(iter.second);
      } else if (iter.first == "trans_b") {
        trans_b = std::get<bool>(iter.second);
      } else if (iter.first == "x_num_col_dims") {
        x_num_col_dims = std::get<int>(iter.second);
      } else if (iter.first == "y_num_col_dims") {
        y_num_col_dims = std::get<int>(iter.second);
      }
    }

    auto out = pe::Matmul(
        A.as_tensor_ref(), B.as_tensor_ref(), trans_a, trans_b, x_num_col_dims, y_num_col_dims, UniqName("C"));
    VLOG(3) << "matmul out: " << out;

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule add_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(add_compute, add_schedule, "strategy.mul.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForMul(const std::vector<shape_t> &inputs_shape, const framework::NodeAttr &attrs) {
  VLOG(3) << "Mul shape0: " << utils::Join(inputs_shape[0], ",");
  VLOG(3) << "Mul shape1: " << utils::Join(inputs_shape[1], ",");
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape is empty";

  int x_num_col_dims = -1;
  if (attrs.attr_store.count("x_num_col_dims")) {
    x_num_col_dims = std::get<int>(attrs.attr_store.at("x_num_col_dims"));
  }
  int y_num_col_dims = -1;
  if (attrs.attr_store.count("y_num_col_dims")) {
    y_num_col_dims = std::get<int>(attrs.attr_store.at("y_num_col_dims"));
  }

  shape_t out_shape;
  for (int i = 0; i < x_num_col_dims; i++) out_shape.push_back(inputs_shape[0][i]);
  for (int i = 0; i < y_num_col_dims; i++) out_shape.push_back(inputs_shape[1][inputs_shape.size() - 1 - i]);

  if (out_shape.empty()) return {{1}};
  return {out_shape};
}

std::vector<Type> InferDtypeForMul(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(transform_ops) {
  CINN_REGISTER_OP(mul)
      .describe("Multiply two tensors")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMul)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForMul))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForMul))
      .set_support_level(4);
  return true;
}

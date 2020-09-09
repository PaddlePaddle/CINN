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
  framework::CINNCompute mul_compute([&attrs](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of mul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for mul compute\n";
    Expr A = a[0];
    Expr B = a[1];
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
      } else {
        LOG(ERROR) << "unsupported attr: " << iter.first << std::endl;
      }
    }

    auto out = pe::Matmul(
        A.as_tensor_ref(), B.as_tensor_ref(), trans_a, trans_b, x_num_col_dims, y_num_col_dims, UniqName("C"));
    VLOG(3) << "matmul out: " << out;

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule mul_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of mul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.mul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMul(const std::vector<std::vector<int>> &inputs_shape,
                                               const framework::NodeAttr &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<int> output_shape;
  std::vector<int> shape1_new;
  std::vector<int> shape2_new;
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
  shape1_new = inputs_shape[0];
  shape2_new = inputs_shape[1];
  if (trans_a) {
    std::reverse(shape1_new.begin(), shape1_new.end());
  }
  if (trans_b) {
    std::reverse(shape2_new.begin(), shape2_new.end());
  }
  output_shape.insert(output_shape.begin(), shape1_new.begin(), shape1_new.begin() + x_num_col_dims);
  output_shape.insert(output_shape.end(), shape2_new.begin() + y_num_col_dims, shape2_new.end());

  if (output_shape.empty()) return {{1}};
  std::vector<std::vector<int>> res{output_shape};
  return res;
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

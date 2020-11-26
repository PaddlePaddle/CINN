#include "cinn/hlir/pe/elementwise.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using namespace pe;
using PeFunc = std::function<ir::Tensor(const ir::Tensor &A, const std::string &out_name)>;

#define StrategyForUnary(op_name__, pe__)                                                            \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                    \
                                                const std::vector<ir::Tensor> &inputs,               \
                                                const std::vector<Type> &out_type,                   \
                                                const std::vector<std::vector<int>> &output_shapes,  \
                                                const Target &target) {                              \
    return StrategyForElementwise(attrs, inputs, out_type, output_shapes, target, #op_name__, pe__); \
  }

std::shared_ptr<OpStrategy> StrategyForElementwise(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target,
                                                   const std::string &op_name,
                                                   const PeFunc &pe_func) {
  framework::CINNCompute unary_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 1U) << "1 input tensor for " << op_name << " compute";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out     = pe_func(A, UniqName(op_name + "_Out"));
    auto stages  = CreateStages({A, out});
    *ret         = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule unary_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    if (target.arch == Target::Arch::NVGPU) {
      Expr Out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(Out.as_tensor());
      pe::CudaSplitSchedule(stages[Out.as_tensor_ref()], output_shapes.back());
      if (Out.as_tensor()->shape.size() > 1) {
        stages[Out.as_tensor_ref()]->Bind(0, "blockIdx.x");
        stages[Out.as_tensor_ref()]->Bind(1, "threadIdx.x");
      }
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(unary_compute, unary_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForElementwise(const std::vector<shape_t> &inputs_shape,
                                              const framework::NodeAttr &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwise(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

StrategyForUnary(exp, Exp);
StrategyForUnary(erf, Erf);
StrategyForUnary(sqrt, Sqrt);
StrategyForUnary(log, Log);
StrategyForUnary(log2, Log2);
StrategyForUnary(log10, Log10);
StrategyForUnary(floor, Floor);
StrategyForUnary(ceil, Ceil);
StrategyForUnary(round, Round);
StrategyForUnary(trunc, Trunc);
StrategyForUnary(cos, Cos);
StrategyForUnary(cosh, Cosh);
StrategyForUnary(tan, Tan);
StrategyForUnary(sin, Sin);
StrategyForUnary(sinh, Sinh);
StrategyForUnary(acos, Acos);
StrategyForUnary(acosh, Acosh);
StrategyForUnary(asin, Asin);
StrategyForUnary(asinh, Asinh);
StrategyForUnary(atan, Atan);
StrategyForUnary(atanh, Atanh);
StrategyForUnary(tanh, Tanh);

StrategyForUnary(isnan, IsNan);
StrategyForUnary(isfinite, IsFinite);
StrategyForUnary(isinf, IsInf);
StrategyForUnary(bitwise_not, BitwiseNot);

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise_ops) {
#define CINN_REGISTER(op__, op_stragegy__)                                                                           \
  CINN_REGISTER_OP(op__)                                                                                             \
      .describe(#op__ " function")                                                                                   \
      .set_num_inputs(1)                                                                                             \
      .set_num_outputs(1)                                                                                            \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__) \
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForElementwise))                               \
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForElementwise))                               \
      .set_support_level(4);

  CINN_REGISTER(exp, Exp);
  CINN_REGISTER(erf, Erf);
  CINN_REGISTER(sqrt, Sqrt);
  CINN_REGISTER(log, Log);
  CINN_REGISTER(log2, Log2);
  CINN_REGISTER(log10, Log10);
  CINN_REGISTER(floor, Floor);
  CINN_REGISTER(ceil, Ceil);
  CINN_REGISTER(round, Round);
  CINN_REGISTER(trunc, Trunc);
  CINN_REGISTER(cos, Cos);
  CINN_REGISTER(cosh, Cosh);
  CINN_REGISTER(tan, Tan);
  CINN_REGISTER(sin, Sin);
  CINN_REGISTER(sinh, Sinh);
  CINN_REGISTER(acos, Acos);
  CINN_REGISTER(acosh, Acosh);
  CINN_REGISTER(asin, Asin);
  CINN_REGISTER(asinh, Asinh);
  CINN_REGISTER(atan, Atan);
  CINN_REGISTER(atanh, Atanh);
  CINN_REGISTER(tanh, Tanh);

  CINN_REGISTER(isnan, IsNan)
  CINN_REGISTER(isfinite, IsFinite)
  CINN_REGISTER(isinf, IsInf)
  CINN_REGISTER(bitwise_not, BitwiseNot)

  return true;
}

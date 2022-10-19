// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/contrib/clip.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "gflags/gflags.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

ir::Tensor Gelu(const ir::Tensor &input, const std::string &output_name) {
  return Compute(
      input->shape,
      [=](const std::vector<Expr> &indice) {
        Expr e = input(indice);
        return e * (make_const(e->type(), 0.5) +
                    make_const(e->type(), 0.5) * lang::Erf(e * make_const(e->type(), std::sqrt(0.5))));
      },
      output_name);
}

std::shared_ptr<OpStrategy> StrategyForGelu(const framework::NodeAttr &attrs,
                                            const std::vector<ir::Tensor> &inputs,
                                            const std::vector<Type> &out_type,
                                            const std::vector<std::vector<int>> &output_shapes,
                                            const Target &target) {
  framework::CINNCompute gelu_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of gelu compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "at least one input tensor for gelu compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    std::string tensor_name = UniqName("Gelu_output");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }
    auto out    = Gelu(A.as_tensor_ref(), tensor_name);
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of gelu op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(gelu_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.gelu.x86", 1);
  } else {
    LOG(FATAL) << "Gelu op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<framework::shape_t> InferShapeForGelu(const std::vector<framework::shape_t> &inputs_shape,
                                                  const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForGelu(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(gelu_ops) {
  CINN_REGISTER_OP(gelu)
      .describe("GELU Activation Operator.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForGelu)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForGelu))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForGelu))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  return true;
}

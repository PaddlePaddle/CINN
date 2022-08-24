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

#include "cinn/hlir/op/contrib/sort.h"

#include <gflags/gflags.h>

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
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor Sort(const ir::Tensor &A, const int &axis, const bool &is_ascend, const std::string &name) {
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        auto out = ir::Store::Make(A, Expr(0), indices);
        return out;
      },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForSort(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  auto attr_store = attrs.attr_store;

  CHECK(attr_store.count("axis")) << "find no attr of axis";
  int axis       = absl::get<int>(attr_store.at("axis"));
  bool is_ascend = true;
  if (attr_store.count("is_ascend")) {
    is_ascend = absl::get<bool>(attr_store.at("is_ascend"));
  }

  framework::CINNCompute sort_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Sort compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 1U) << "At least 1 input tensors for Sort compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto stages   = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    ir::Tensor out = Sort(tensor_A, axis, is_ascend, UniqName("Sort_out"));
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Sort is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule sort_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of reshape schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    Expr out               = arg_pack[0];
    CHECK(out.as_tensor());
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(sort_compute, sort_schedule, "strategy.sort.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForSort(const std::vector<std::vector<int>> &inputs_shape,
                                                const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForSort(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(cast_ops) {
  CINN_REGISTER_OP(cast)
      .describe("Sort.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSort)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSort))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSort))
      .set_support_level(4);

  return true;
}

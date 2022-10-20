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

#include "cinn/hlir/op/contrib/lookup_table.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "gflags/gflags.h"
DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor LookupTable(const ir::Tensor& table,
                       const ir::Tensor& ids,
                       const int64_t padding_idx,
                       const std::string& output_name) {
  CHECK_EQ(table->shape.size(), 2);
  CHECK_GT(ids->shape.size(), 1);
  auto output_shape   = ids->shape;
  output_shape.back() = table->shape.back();

  return lang::Compute(
      output_shape,
      [&](const std::vector<ir::Expr>& indices) {
        Expr id{1};
        std::vector<Expr> offsets;
        for (int i = 0; i < indices.size() - 1; ++i) {
          id = id * indices[i];
          offsets.emplace_back(indices[i]);
        }
        offsets.emplace_back(Expr(0));
        // Because the current conversion rules have not been completed, static conversion is done here.
        auto pred       = Expr(padding_idx != -1 && id == Expr(static_cast<int32_t>(padding_idx)));
        auto ids_offset = ir::Cast::Make(common::I32(), ids(offsets));
        return ir::Select::Make(pred, common::make_const(table->type(), 0), table(ids_offset, indices.back()));
      },
      common::UniqName(output_name));
}

std::shared_ptr<framework::OpStrategy> StrategyForLookupTable(const framework::NodeAttr& attrs,
                                                              const std::vector<ir::Tensor>& inputs,
                                                              const std::vector<Type>& out_type,
                                                              const std::vector<std::vector<int>>& output_shapes,
                                                              const Target& target) {
  std::string op_name("lookup_table");
  const auto& attr_store = attrs.attr_store;
  CHECK(attr_store.count("padding_idx")) << "find no attr of axis";
  auto padding_idx = absl::get<int64_t>(attr_store.at("padding_idx"));

  framework::CINNCompute lookup_table_compute([=](lang::Args args, lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input arguments of " << op_name << " compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U) << "2 input tensors for " << op_name << " compute\n";
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages   = CreateStages({tensor_A, tensor_B});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ") << ", B shape: " << utils::Join(tensor_B->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    std::string tensor_name = UniqName("LookupTable_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 3U);
      tensor_name = pack_args[2].operator std::string();
    }
    ir::Tensor out = LookupTable(tensor_A, tensor_B, padding_idx, tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule lookup_table_schedule([=](lang::Args args, lang::RetValue* ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of lookup_table_schedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      std::vector<Expr> vec_ast;
      for (int i = 0; i < arg_pack.size(); i++) {
        if (arg_pack[i].is_expr()) {
          Expr temp = arg_pack[i];
          vec_ast.emplace_back(temp);
        }
      }
      CHECK(!vec_ast.empty());
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      ir_sch.MergeExprs();
      long prod_size = std::accumulate(output_shapes[0].begin(), output_shapes[0].end(), 1, std::multiplies<int>());
      if (prod_size > 1) {
        if (target.arch == Target::Arch::NVGPU) {
          pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
        } else if (target.arch == Target::Arch::X86) {
          pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
        }
      }
      std::vector<common::CINNValue> res{common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = common::CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input argument of lookup_table_schedule is empty! Please check.\n";
      CINNValuePack arg_pack = args[0];
      Expr out               = arg_pack[0];
      CHECK(out.as_tensor());
      *ret = arg_pack;
    }
  });
  auto strategy = std::make_shared<framework::OpStrategy>();
  if (target.arch == Target::Arch::NVGPU) {
    strategy->AddImpl(lookup_table_compute, lookup_table_schedule, "strategy.lookup_table.cuda", 1);
  } else {
    strategy->AddImpl(lookup_table_compute, lookup_table_schedule, "strategy.lookup_table.x86", 1);
  }
  return strategy;
}

std::vector<framework::shape_t> InferShapeForLookupTable(const std::vector<framework::shape_t>& inputs_shape,
                                                         const framework::AttrMapType& attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";

  auto res   = inputs_shape[1];
  res.back() = inputs_shape[0].back();
  return {res};
}

std::vector<Type> InferDtypeForLookupTable(const std::vector<Type>& inputs_type, const framework::AttrMapType& attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(lookup_table_ops) {
  CINN_REGISTER_OP(lookup_table)
      .describe("Lookup table Operator.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForLookupTable)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForLookupTable))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForLookupTable))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible);
  return true;
}

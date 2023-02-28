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
#include "cinn/hlir/op/op_util.h"
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
        std::vector<Expr> offsets;
        for (int i = 0; i < indices.size() - 1; ++i) {
          offsets.emplace_back(indices[i]);
        }
        offsets.emplace_back(Expr(0));
        // Because the current conversion rules have not been completed, static conversion is done here.
        auto ids_offset = ir::Cast::Make(common::I32(), ids(offsets));
        auto pred =
            ir::And::Make(Expr(padding_idx != -1), ir::EQ::Make(ids_offset, Expr(static_cast<int32_t>(padding_idx))));
        return ir::Select::Make(pred, ir::Cast::Make(table->type(), Expr(0)), table(ids_offset, indices.back()));
      },
      common::UniqName(output_name));
}

CINNSchedule ScheduleFunc(const std::vector<std::vector<int>>& output_shapes, const Target& target) {
  // Configure the schedule for intermediate results
  return CINNSchedule([=](lang::Args args, lang::RetValue* ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of InjectiveSchedule is empty! Please check.\n";
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
      VLOG(3) << "Before IRInjectiveSchedule, new ir is : " << ir_sch.GetModule().GetExprs().at(0);
      if (target == common::DefaultNVGPUTarget()) {
        auto blocks       = ir_sch.GetAllBlocks();
        auto output_shape = output_shapes.front();
        for (size_t i = 0; i < blocks.size(); ++i) {
          if (i < blocks.size() - 1) {
            // CUDA device codegen not support memory type heap
            ir_sch.SetBuffer(blocks[i], "local");
          }
          ir_sch.FlattenLoops(ir_sch.GetLoops(blocks[i]), false);
          auto loops = ir_sch.GetLoops(blocks[i]);
          auto size  = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
          if (size <= target.max_num_threads()) {
            ir_sch.Bind(loops[0], "threadIdx.x");
          } else {
            auto splited = ir_sch.Split(loops[0], {-1, target.max_num_threads() / 4});
            ir_sch.Bind(splited[0], "blockIdx.x");
            ir_sch.Bind(splited[1], "threadIdx.x");
          }
        }
      } else {
        LOG(FATAL) << "unsupported scheduler.";
      }
      VLOG(3) << "After IRInjectiveSchedule, new ir is : " << ir_sch.GetModule().GetExprs().at(0);
      std::vector<common::CINNValue> res{common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = common::CINNValuePack{res};
    } else {
      LOG(FATAL) << "unsupported scheduler.";
    }
  });
}

std::shared_ptr<framework::OpStrategy> StrategyForLookupTable(const framework::NodeAttr& attrs,
                                                              const std::vector<ir::Tensor>& inputs,
                                                              const std::vector<Type>& out_type,
                                                              const std::vector<std::vector<int>>& output_shapes,
                                                              const Target& target) {
  std::string op_name("lookup_table");
  const auto& attr_store = attrs.attr_store;
  CHECK(attr_store.count("padding_idx")) << "find no attr of axis";
  auto padding_idx = absl::get<int32_t>(attr_store.at("padding_idx"));

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
    ir::Tensor out_inplace = LookupTable(tensor_A, tensor_B, padding_idx, tensor_name);

    ir::Tensor out = lang::Compute(
        out_inplace->shape, [&](const std::vector<ir::Expr>& indices) { return out_inplace(indices); }, "Assign__");

    std::vector<CINNValue> res;
    stages->InsertLazily(out_inplace);
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(lookup_table_compute, ScheduleFunc(output_shapes, target), "strategy.lookup_table", 1);
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

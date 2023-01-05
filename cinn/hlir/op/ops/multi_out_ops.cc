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

#include "absl/types/optional.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {

std::shared_ptr<OpStrategy> StrategyForSplit(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const std::vector<std::vector<int>> &output_shapes,
                                             const Target &target) {
  // get attribute
  std::vector<int> sections;
  int axis = 0;
  if (attrs.attr_store.find("num_or_sections") != attrs.attr_store.end()) {
    sections = absl::get<std::vector<int>>(attrs.attr_store.at("num_or_sections"));
  }
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  if (axis < 0) axis += static_cast<int>(output_shapes[0].size());

  CHECK(!output_shapes.empty()) << "The Spilt Op's output shape list should not empty.";
  CHECK_LT(axis, static_cast<int>(output_shapes[0].size()));
  CHECK(!sections.empty())
      << "The Split op doesn't find [num_or_sections] attrbute! It it a mandatory attribute ! Please check.";

  framework::CINNCompute split_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of split compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "The input tensors of split compute is empty! Please check.";
    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();

    std::vector<std::string> tensor_names;
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), output_shapes.size() + 1);
      for (int idx = 1; idx < pack_args.size(); ++idx) {
        CHECK(pack_args[idx].is_string());
        tensor_names.push_back(pack_args[idx].operator std::string());
      }
    } else {
      for (int idx = 0; idx < output_shapes.size(); ++idx) {
        tensor_names.push_back(UniqName("T_Split_Out"));
      }
    }

    auto out    = pe::Split(A, axis, output_shapes, tensor_names);
    auto stages = CreateStages(out);

    std::vector<CINNValue> res;
    for (int i = 0; i < out.size(); ++i) {
      res.emplace_back(out[i]);
    }
    res.emplace_back(stages);
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule split_schedule([=](lang::Args args, lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of split schedule is empty! Please check.";
      CINNValuePack arg_pack = args[0];
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
      pe::IRCudaSplitSchedule(ir_sch, output_shapes, axis, target);
      std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input arguments of split schedule is empty! Please check.";
      CINNValuePack arg_pack = args[0];
      CHECK_GE(arg_pack.size(), 2UL) << "The input tensor's size of split schedule is " << arg_pack.size()
                                     << "and it should be greater equal to 2! Please check.";
      pe::CudaSplitSchedule(&arg_pack, output_shapes, axis, target);
      *ret = arg_pack;
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(split_compute, split_schedule, "strategy.split.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForSplit(const std::vector<std::vector<int>> &inputs_shape,
                                                 const framework::AttrMapType &attrs) {
  std::vector<int> sections;
  if (attrs.find("num_or_sections") != attrs.end()) {
    sections = absl::get<std::vector<int>>(attrs.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it a mandatory attribute ! Please check.";
  }

  if (inputs_shape.empty()) {
    std::vector<std::vector<int>> ret;
    if (sections.size() == 1) {
      ret.resize(sections[0]);
    } else {
      ret.resize(sections.size());
    }
    return ret;
  }
  CHECK_GE(inputs_shape.size(), 1U) << "The input's shape size should be no less than 1! Please check again.";

  int axis = 0;
  if (attrs.find("axis") != attrs.end()) {
    axis = absl::get<int>(attrs.at("axis"));
    if (axis < 0) {
      axis += inputs_shape[0].size();
    }
  }

  // check sections valid
  int output_size = sections.size();
  int pivot       = inputs_shape[0][axis];

  auto real_sections = sections;
  if (output_size == 1) {
    // if the 'sections' is a number, the tensor will split to 'sections' sub-tensor, each sub-tensor length A[axis] /
    // 'sections'
    output_size = sections[0];
    CHECK_EQ(pivot % output_size, 0) << "If the attribute 'num_or_sections' is a number, it should be divisible by the "
                                        "axis's dimension of inputs A ! Please check.";
    real_sections.assign(output_size, pivot / output_size);
  } else {
    // else the tensor will split to sections.size sub-tensor, each sub-tensor length sections[i]
    // The sections may have at most one '-1' in sections, that means its value should be inferred by others.
    int section_sum = 0, neg_index = -1;
    for (int i = 0; i < output_size; ++i) {
      if (sections[i] > 0) {
        section_sum += sections[i];
      } else if (sections[i] == -1 && neg_index < 0) {
        neg_index = i;
      } else {
        if (sections[i] == 0) {
          LOG(FATAL) << "The attribute 'num_or_sections' should not has 0 ! Please check.";
        } else {
          LOG(FATAL) << "The attribute 'num_or_sections' can only have at most one '-1' ! Please check.";
        }
      }
    }

    if (neg_index >= 0) {
      // has '-1' in sections
      real_sections[neg_index] = pivot - section_sum;
    } else {
      CHECK_EQ(pivot, section_sum) << "The sum of attr sections should be equal with the axis's dimension value of "
                                      "inputs A in Split ! Please check.";
    }
  }

  std::vector<std::vector<int>> outputs_shape(output_size, inputs_shape[0]);
  for (int i = 0; i < output_size; ++i) {
    outputs_shape[i][axis] = real_sections[i];
  }
  return outputs_shape;
}

std::vector<Type> InferDtypeForSplit(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";

  std::vector<int> sections;
  if (attrs.find("num_or_sections") != attrs.end()) {
    sections = absl::get<std::vector<int>>(attrs.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it a mandatory attribute ! Please check.";
  }

  int output_size = sections.size();
  if (output_size == 1) {
    output_size = sections[0];
  }

  std::vector<Type> res(output_size, inputs_type[0]);
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForSplit(const std::vector<framework::shape_t> &input_shapes,
                                                          const std::vector<std::string> &input_layouts,
                                                          const framework::NodeAttr &attrs,
                                                          const Target &target) {
  CHECK(!input_layouts.empty()) << "The input's layout size is 0! Please check again.";
  std::vector<int> sections;
  if (attrs.attr_store.find("num_or_sections") != attrs.attr_store.end()) {
    sections = absl::get<std::vector<int>>(attrs.attr_store.at("num_or_sections"));
  } else {
    LOG(FATAL) << "The Split op doesn't find [num_or_sections] attrbute! It it a mandatory attribute ! Please check.";
  }

  int output_size = sections.size();
  if (output_size == 1) {
    output_size = sections[0];
  }

  std::vector<std::string> output_layout(output_size, input_layouts[0]);
  return {output_layout, input_layouts};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

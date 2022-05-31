// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <numeric>

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace science_mappers {

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("XS").size(), 1UL);
  auto x_names = op_desc.Input("XS");
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  Variable out;
  if (x_names.size() == 1) {
    // if concat only has one input, using Identity to copy the input and return
    auto x = ctx.GetVar(x_names.front());
    out    = ctx.Builder()->Identity(x);
  } else {
    std::vector<Variable> xs;
    for (const auto& name : x_names) {
      xs.emplace_back(ctx.GetVar(name));
    }

    auto axis = utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

    out = ctx.Builder()->Concat(xs, axis);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SplitOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_GE(op_desc.Output("YS").size(), 1UL);
  auto out_name = op_desc.Output("YS");

  CHECK(op_desc.HasAttr("num_or_sections"));
  auto num_or_sections = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("num_or_sections"));

  CHECK(!num_or_sections.empty()) << "The Split op cannot found [num_or_sections] attrbute!  ! Please check.";

  auto axis = utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);

  auto x_shape = x->shape;
  if (num_or_sections.size() == 1U) {
    CHECK_EQ(x_shape[axis] % num_or_sections[0], 0)
        << "If the attribute 'num_or_sections' is a number, it should be divisible by the "
           "axis's dimension of inputs A ! Please check.";
  } else {
    cinn::utils::DimType sec_sum = 0;
    bool has_neg                 = false;
    for (auto sec : num_or_sections) {
      if (sec > 0) {
        sec_sum += sec;
      } else if (sec == -1 && !has_neg) {
        has_neg = true;
      } else if (sec == 0) {
        LOG(FATAL) << "The attribute 'num_or_sections' of split should not has 0 ! Please check.";
      } else {
        LOG(FATAL) << "The attribute 'num_or_sections' of split can only have at most one '-1' ! Please check.";
      }
    }
    CHECK(!has_neg && sec_sum == x_shape[axis])
        << "The sum of attr sections should be equal with the axis's dimension value of "
           "inputs A in Split ! Please check.";
  }

  VLOG(4) << "Split " << x_name << " with shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << " to section (" << cinn::utils::Join(num_or_sections, ",") << ") at dimension " << axis;

  auto out = ctx.Builder()->Split(x, num_or_sections, axis);

  CHECK_EQ(out.size(), out_name.size()) << "The Split op should has " << out_name.size() << " output, but only "
                                        << out.size();

  for (int i = 0; i < out.size(); ++i) {
    ctx.AddVar(out_name[i], out[i]);
    ctx.AddVarModelToProgram(out_name[i], out[i]->id);
  }
}

void ReshapeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  auto shape = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape"));

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reshape " << x_name << "from shape (" << cinn::utils::Join(x->shape, ",") << ") to ("
          << cinn::utils::Join(shape, ",") << ").";

  auto out = ctx.Builder()->Reshape(x, shape);

  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void TransposeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  CHECK(x->shape.size() == 2) << "Now transpose_p only support 2-dim matrix.";
  VLOG(4) << "Transpose " << x_name << " with shape (" << cinn::utils::Join(x->shape, ",") << ").";

  auto out = ctx.Builder()->Transpose(x, {1, 0});

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SliceSelectOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  CHECK(op_desc.HasAttr("starts"));
  auto starts = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("starts"));
  CHECK(op_desc.HasAttr("ends"));
  auto ends = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("ends"));
  CHECK(op_desc.HasAttr("axis"));
  auto axes = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("axis"));
  CHECK(op_desc.HasAttr("strides"));
  auto strides = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "strides"));

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "SliceSelect " << x_name << " from shape (" << cinn::utils::Join(x->shape, ",") << ") with starts ["
          << cinn::utils::Join(starts, ",") << "], ends [" << cinn::utils::Join(ends, ",") << "], axis ["
          << cinn::utils::Join(axes, ",") << "], strides [" << cinn::utils::Join(strides, ",") << "].";

  auto out = ctx.Builder()->Slice(x, axes, starts, ends, cinn::utils::ShapeType{}, strides);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SliceAssignOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  CHECK(op_desc.HasAttr("starts"));
  auto starts = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("starts"));
  CHECK(op_desc.HasAttr("ends"));
  auto ends = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("ends"));
  CHECK(op_desc.HasAttr("axis"));
  auto axes = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("axis"));
  CHECK(op_desc.HasAttr("strides"));
  auto strides = utils::ToShapeType(op_desc.GetAttr<std::vector<int64_t>>("strides"));

  auto x      = ctx.GetVar(x_name);
  auto assign = ctx.GetVar(y_name);

  VLOG(4) << "SliceAssign " << x_name << " from shape (" << cinn::utils::Join(x->shape, ",") << ") with starts ["
          << cinn::utils::Join(starts, ",") << "], ends [" << cinn::utils::Join(ends, ",") << "], axis ["
          << cinn::utils::Join(axes, ",") << "], strides [" << cinn::utils::Join(strides, ",") << "].";

  auto out = ctx.Builder()->SliceAssign(x, assign, axes, starts, ends, strides);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ReduceOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto axis    = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "axis"));
  auto keepdim = utils::GetAttrOrDefault<bool>(op_desc, "keepdim", false);

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reudce " << x_name << " from shape (" << cinn::utils::Join(x->shape, ",") << "), with axis "
          << cinn::utils::Join(axis, ",") << ", keepdim " << keepdim;

  // now paddle science only need reduce sum
  Variable out;
  if (std::accumulate(x->shape.begin(), x->shape.end(), 1, std::multiplies<cinn::utils::DimType>()) == 1) {
    out = ctx.Builder()->Identity(x);
    if (!keepdim) {
      cinn::utils::ShapeType new_out_shape;
      for (int i = 0; i < x->shape.size(); ++i) {
        if (std::find(axis.begin(), axis.end(), static_cast<cinn::utils::DimType>(i)) == axis.end()) {
          new_out_shape.emplace_back(x->shape[i]);
        }
      }
      out->shape = new_out_shape;
    }
  } else {
    out = ctx.Builder()->ReduceSum(x, axis, keepdim);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void IndexSelectOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("IndexTensor").size(), 1UL);
  auto index_name = op_desc.Input("IndexTensor").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto axis = utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x     = ctx.GetVar(x_name);
  auto index = ctx.GetVar(index_name);

  VLOG(4) << "IndexSelect " << index_name << " (" << cinn::utils::Join(index->shape, ",") << ") from " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  auto out = ctx.Builder()->IndexSelect(x, index, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void IndexAssignOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto updates_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Input("IndexTensor").size(), 1UL);
  auto index_name = op_desc.Input("IndexTensor").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  auto axis = utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x       = ctx.GetVar(x_name);
  auto updates = ctx.GetVar(updates_name);
  auto index   = ctx.GetVar(index_name);

  auto out = ctx.Builder()->ScatterAssign(x, updates, index, axis);

  VLOG(4) << "IndexAssign " << updates_name << " (" << cinn::utils::Join(updates->shape, ",") << ") to " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ScatterAddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto updates_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Input("IndexTensor").size(), 1UL);
  auto index_name = op_desc.Input("IndexTensor").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  auto axis = utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x       = ctx.GetVar(x_name);
  auto updates = ctx.GetVar(updates_name);
  auto index   = ctx.GetVar(index_name);

  auto out = ctx.Builder()->ScatterAdd(x, updates, index, axis);

  VLOG(4) << "ScatterAdd " << updates_name << " (" << cinn::utils::Join(updates->shape, ",") << ") to " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_transform) {
  CINN_REGISTER_OP_MAPPER(concat_p, cinn::frontend::science_mappers::ConcatOpMapper)
  CINN_REGISTER_OP_MAPPER(split_p, cinn::frontend::science_mappers::SplitOpMapper)
  CINN_REGISTER_OP_MAPPER(reshape_p, cinn::frontend::science_mappers::ReshapeOpMapper)
  CINN_REGISTER_OP_MAPPER(transpose_p, cinn::frontend::science_mappers::TransposeOpMapper)
  CINN_REGISTER_OP_MAPPER(slice_select_p, cinn::frontend::science_mappers::SliceSelectOpMapper)
  CINN_REGISTER_OP_MAPPER(slice_assign_p, cinn::frontend::science_mappers::SliceAssignOpMapper)
  CINN_REGISTER_OP_MAPPER(index_select_p, cinn::frontend::science_mappers::IndexSelectOpMapper)
  CINN_REGISTER_OP_MAPPER(gather_p, cinn::frontend::science_mappers::IndexSelectOpMapper)
  CINN_REGISTER_OP_MAPPER(index_assign_p, cinn::frontend::science_mappers::IndexAssignOpMapper)
  CINN_REGISTER_OP_MAPPER(scatter_add_p, cinn::frontend::science_mappers::ScatterAddOpMapper)
  CINN_REGISTER_OP_MAPPER(reduce_p, cinn::frontend::science_mappers::ReduceOpMapper)
  return true;
}

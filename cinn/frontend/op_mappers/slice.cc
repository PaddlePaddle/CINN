#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void slice_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("starts"));
  auto starts = op_desc.GetAttr<std::vector<int>>("starts");
  CHECK(op_desc.HasAttr("ends"));
  auto ends = op_desc.GetAttr<std::vector<int>>("ends");
  CHECK(op_desc.HasAttr("axes"));
  auto axes = op_desc.GetAttr<std::vector<int>>("axes");

  auto infer_flags   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "infer_flags");
  auto decrease_axis = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "decrease_axis");
  auto x             = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out           = ctx.builder_->slice(x, axes, starts, ends, infer_flags, decrease_axis);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_OPMAPPER(slice, cinn::frontend::op_mappers::slice_kernel)

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void add_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x   = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto y   = utils::GetVar(cinn::utils::TransValidVarName(y_name), ctx);
  auto out = ctx.builder_->add(x, y);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

void elementwise_add_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x   = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto y   = utils::GetVar(cinn::utils::TransValidVarName(y_name), ctx);
  auto out = ctx.builder_->elementwise_add(x, y, axis);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

void elementwise_mul_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x   = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto y   = utils::GetVar(cinn::utils::TransValidVarName(y_name), ctx);
  auto out = ctx.builder_->elementwise_mul(x, y, axis);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise) {
  CINN_REGISTER_OPMAPPER(add, cinn::frontend::op_mappers::add_kernel)
  CINN_REGISTER_OPMAPPER(elementwise_add, cinn::frontend::op_mappers::elementwise_add_kernel)
  CINN_REGISTER_OPMAPPER(elementwise_mul, cinn::frontend::op_mappers::elementwise_mul_kernel)
}

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void relu_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto x        = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out      = ctx.builder_->relu(x);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

void relu6_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto threshold = utils::GetAttrOrDefault<float>(op_desc, "threshold", 6.0f);
  auto x         = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out       = ctx.builder_->relu6(x, threshold);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_OPMAPPER(relu, cinn::frontend::op_mappers::relu_kernel)
CINN_REGISTER_OPMAPPER(relu6, cinn::frontend::op_mappers::relu6_kernel)

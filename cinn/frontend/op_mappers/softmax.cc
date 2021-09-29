#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void softmax_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis        = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);
  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");

  auto x   = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out = ctx.builder_->softmax(x, axis, data_format);
  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(softmax) { CINN_REGISTER_OPMAPPER(softmax, cinn::frontend::op_mappers::softmax_kernel) }

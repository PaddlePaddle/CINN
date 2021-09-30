#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void DropoutInferKernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto dropout_prob = utils::GetAttrOrDefault<float>(op_desc, "dropout_prob", 0.5f);
  auto dropout_implementation =
      utils::GetAttrOrDefault<std::string>(op_desc, "dropout_implementation", "downgrade_in_infer");
  auto x   = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out = ctx.builder_->dropout_infer(x, dropout_prob, dropout_implementation);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(dropout) { CINN_REGISTER_OP_MAPPER(dropout, cinn::frontend::op_mappers::DropoutInferKernel) }

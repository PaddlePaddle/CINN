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
  auto x   = ctx.GetVar(x_name);
  auto out = ctx.builder_->dropout_infer(x, dropout_prob, dropout_implementation);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(dropout) { CINN_REGISTER_OP_MAPPER(dropout, cinn::frontend::op_mappers::DropoutInferKernel) }

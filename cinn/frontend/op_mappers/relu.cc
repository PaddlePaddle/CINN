#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void ReluOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto x        = ctx.GetVar(x_name);
  auto out      = ctx.builder_->relu(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

void Relu6OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto threshold = utils::GetAttrOrDefault<float>(op_desc, "threshold", 6.0f);
  auto x         = ctx.GetVar(x_name);
  auto out       = ctx.builder_->relu6(x, threshold);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(relu) {
  CINN_REGISTER_OP_MAPPER(relu, cinn::frontend::op_mappers::ReluOpMapper)
  CINN_REGISTER_OP_MAPPER(relu6, cinn::frontend::op_mappers::Relu6OpMapper)
  return true;
}

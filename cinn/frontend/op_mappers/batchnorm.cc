#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void batchnorm_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Scale").size(), 1UL);
  auto scale_name = op_desc.Input("Scale").front();
  CHECK_EQ(op_desc.Input("Bias").size(), 1UL);
  auto bias_name = op_desc.Input("Bias").front();
  CHECK_EQ(op_desc.Input("Mean").size(), 1UL);
  auto mean_name = op_desc.Input("Mean").front();
  CHECK_EQ(op_desc.Input("Variance").size(), 1UL);
  auto variance_name = op_desc.Input("Variance").front();
  CHECK(!op_desc.Output("Y").empty());
  auto out_name = op_desc.Output("Y").front();

  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);
  auto momentum    = utils::GetAttrOrDefault<float>(op_desc, "momentum", 0.9f);
  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto x           = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto scale       = utils::GetVar(cinn::utils::TransValidVarName(scale_name), ctx);
  auto bias        = utils::GetVar(cinn::utils::TransValidVarName(bias_name), ctx);
  auto mean        = utils::GetVar(cinn::utils::TransValidVarName(mean_name), ctx);
  auto variance    = utils::GetVar(cinn::utils::TransValidVarName(variance_name), ctx);
  auto out         = ctx.builder_->batchnorm(x, scale, bias, mean, variance, epsilon, momentum, data_layout);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batchnorm) { CINN_REGISTER_OPMAPPER(batchnorm, cinn::frontend::op_mappers::batchnorm_kernel) }

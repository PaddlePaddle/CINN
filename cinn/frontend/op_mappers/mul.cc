#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void mul_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  auto x      = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  utils::TransposeVar(cinn::utils::TransValidVarName(y_name), ctx);
  auto y = utils::GetVar(cinn::utils::TransValidVarName(y_name), ctx);

  auto x_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "x_num_col_dims", 1);
  auto y_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "y_num_col_dims", 1);

  VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
  VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "y shape: " << cinn::utils::Join(y->shape, ",");
  auto out = ctx.builder_->mul(x, y, x_num_col_dims, y_num_col_dims);
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_OPMAPPER(mul, cinn::frontend::op_mappers::mul_kernel)

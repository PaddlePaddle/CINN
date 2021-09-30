#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void Pool2dKernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("pooling_type"));
  auto pooling_type = op_desc.GetAttr<std::string>("pooling_type");
  CHECK(op_desc.HasAttr("ksize"));
  auto ksize = op_desc.GetAttr<std::vector<int>>("ksize");

  auto strides      = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto padding_size = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});

  if (padding_size.size() == 2) {
    padding_size.insert(padding_size.begin(), padding_size.front());
    padding_size.push_back(padding_size.back());
  }

  auto ceil_mode         = utils::GetAttrOrDefault<bool>(op_desc, "ceil_mode", false);
  auto exclusive         = utils::GetAttrOrDefault<bool>(op_desc, "exclusive", true);
  auto global_pooling    = utils::GetAttrOrDefault<bool>(op_desc, "global_pooling", false);
  auto data_format       = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  auto adaptive          = utils::GetAttrOrDefault<bool>(op_desc, "adaptive", false);
  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  auto out               = ctx.builder_->pool2d(x,
                                  pooling_type,
                                  ksize,
                                  strides,
                                  padding_size,
                                  ceil_mode,
                                  exclusive,
                                  global_pooling,
                                  data_format,
                                  adaptive,
                                  padding_algorithm);

  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(pool2d) { CINN_REGISTER_OP_MAPPER(pool2d, cinn::frontend::op_mappers::Pool2dKernel) }

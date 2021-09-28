#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void fetch_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto output_name = op_desc.Input("X").front();
  LOG(INFO) << "detect model output: [" << output_name << "]";
}

void feed_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto outs = op_desc.Output("Out");
  CHECK_EQ(outs.size(), 1UL);
  VLOG(2) << "Model get feed [" << outs[0] << "]";
  Placeholder input(common::Float(32), {}, outs[0]);
  utils::AddVar(outs[0], input, ctx);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_OPMAPPER(fetch, cinn::frontend::op_mappers::fetch_kernel)
CINN_REGISTER_OPMAPPER(feed, cinn::frontend::op_mappers::feed_kernel)

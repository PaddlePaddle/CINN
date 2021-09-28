#include "cinn/frontend/paddle_model_to_netbuilder.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/runtime/use_extern_funcs.h"

DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

TEST(PaddleModelToNetBuilder, basic) {
  auto scope  = hlir::framework::Scope::Create();
  auto target = common::DefaultHostTarget();

  PaddleModelToNetBuilder model_transform(scope.get(), target);
  auto builder = model_transform(FLAGS_model_dir);

  const auto& var_map                  = model_transform.var_map();
  const auto& var_model_to_program_map = model_transform.var_model_to_program_map();

  ASSERT_FALSE(var_map.empty());
  ASSERT_FALSE(var_model_to_program_map.empty());
  LOG(INFO) << builder->name();
}

}  // namespace frontend
}  // namespace cinn

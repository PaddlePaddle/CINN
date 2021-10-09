#include "cinn/frontend/op_mapper_registry.h"

#include <gtest/gtest.h>

#include <typeinfo>

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/utils/registry.h"

namespace cinn {
namespace frontend {

TEST(OpMapperRegistryTest, basic) {
  auto kernel = OpMapperRegistry::Global()->Find("sigmoid");
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(typeid(*kernel), typeid(OpMapper));
  ASSERT_EQ(kernel->name, "sigmoid");
}

}  // namespace frontend
}  // namespace cinn

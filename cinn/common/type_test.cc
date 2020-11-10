#include "cinn/common/type.h"

#include <gtest/gtest.h>

namespace cinn::common {

TEST(Type, basic) {
  LOG(INFO) << I32();

  auto i32 = I32();
  LOG(INFO) << I32();

  LOG(INFO) << F32();
  LOG(INFO) << type_of<float>();
}

}  // namespace cinn::common

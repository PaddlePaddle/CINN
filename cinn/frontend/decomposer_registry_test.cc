#include "cinn/frontend/decomposer_registry.h"

#include <gtest/gtest.h>

namespace cinn::frontend {

TEST(InstrDecomposerRegistry, basic) {
  common::Target target = common::DefaultHostTarget();
  ASSERT_NE(Decomposer::Global()->Find("conv", target), nullptr);
}

}  // namespace cinn::frontend

#include "cinn/frontend/decomposer_registry.h"

#include <gtest/gtest.h>

#include "cinn/frontend/decomposer/use_decomposer.h"

namespace cinn::frontend {

TEST(InstrDecomposerRegistry, basic) {
  common::Target target = common::DefaultHostTarget();
  ASSERT_NE(InstrDecomposerRegistry::Global()->Find("conv", target), nullptr);
}

}  // namespace cinn::frontend

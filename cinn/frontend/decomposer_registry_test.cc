#include "cinn/frontend/decomposer_registry.h"

#include <gtest/gtest.h>

namespace cinn::frontend {

TEST(InstrDecomposerRegistry, basic) {
  common::Target target;
  auto decomposer =
      [](const Instruction& instr, Program* program, std::unordered_map<std::string, Variable>* outs_map) {
        auto var           = frontend::Variable("var");
        (*outs_map)["var"] = var;
      };

  CINN_REGISTER_INSTR_DECOMPOSER("test", target, decomposer);

  ASSERT_EQ(InstrDecomposerMap::Instance().Has("test", target), true);
  InstrDecomposerMap::Instance().Get("test", target);
}

}  // namespace cinn::frontend

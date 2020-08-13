#include "cinn/hlir/framework/graph_compiler.h"

#include "cinn/hlir/framework/instruction.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<Program> GraphCompiler::Build() {
  auto [nodes, edges] = graph_->topological_order();
  for (auto n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node->op());
      m_builder_.AddFunction(lowered_func);
    }
  }

  LOG(INFO) << "Compile the module";
  // compile the module
  CHECK(compiler_);
  compiler_->Build(m_builder_.Build());

  return std::unique_ptr<Program>(new Program(scope_, BuildInstructions()));
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions() {
  std::vector<std::unique_ptr<Instruction>> instructions;

  auto [nodes, edges] = graph_->topological_order();
  for (auto n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), OpGetInputNames(node->op()), OpGetOutputNames(node->op())));
      auto* fn = compiler_->Lookup(GenOpFuncName(node->op()));
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      instructions.push_back(std::move(instr));
    }
  }
  return instructions;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

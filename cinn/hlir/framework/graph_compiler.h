#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/compiler.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"

namespace cinn {
namespace hlir {
namespace framework {

class Program {
 public:
  Program(const std::shared_ptr<Scope>& scope, std::vector<std::unique_ptr<Instruction>>&& instrs)
      : scope_(scope), instrs_(std::move(instrs)) {}

  void Execute() {
    for (auto& ins : instrs_) ins->Run();
  }

  size_t size() const { return instrs_.size(); }

 private:
  // We need to hold scope to assure tensors alive used in instructions.
  std::shared_ptr<Scope> scope_;
  std::vector<std::unique_ptr<Instruction>> instrs_;
};

class GraphCompiler final {
 public:
  GraphCompiler(Target target, const std::shared_ptr<Scope>& scope, Graph* const graph)
      : target_(std::move(target)), scope_(scope), graph_(graph), m_builder_(UniqName("module"), target) {}

  std::unique_ptr<Program> Build();

 private:
  ir::LoweredFunc GetOpFunc(const Node* node);

  std::string GenOpFuncName(const Node* node) const {
    return "fn_" + node->op()->name + "_" + std::to_string(node->op()->get_index());
  }

  // TODO(haozech) add implementation
  std::vector<std::string> OpGetInputNames(const Node* node) const;
  // TODO(haozech) add implementation
  std::vector<std::string> OpGetOutputNames(const Node* node) const;

  std::vector<std::unique_ptr<Instruction>> BuildInstructions();

 private:
  Target target_;
  Graph* const graph_{};
  std::shared_ptr<Scope> scope_;

  std::unique_ptr<backends::Compiler> compiler_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);

  lang::Module::Builder m_builder_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

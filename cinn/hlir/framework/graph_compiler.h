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
#include "cinn/ir/packed_func.h"

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
      : target_(std::move(target)),
        scope_(scope),
        graph_(graph),
        m_builder_(Context::Global().NewName("module"), target) {}

  std::unique_ptr<Program> Build();

 private:
  // TODO(haozech) add implementation
  ir::LoweredFunc GetOpFunc(const Operator* op) { CINN_NOT_IMPLEMENTED; }

  std::string GenOpFuncName(const Operator* op) const {
    return "fn_" + op->name + "_" + std::to_string(op->get_index());
  }

  // TODO(haozech) add implementation
  std::vector<std::string> OpGetInputNames(const Operator* op) const { CINN_NOT_IMPLEMENTED; }
  // TODO(haozech) add implementation
  std::vector<std::string> OpGetOutputNames(const Operator* op) const { CINN_NOT_IMPLEMENTED; }

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

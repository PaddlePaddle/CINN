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

/**
 * The Program is the runtime instance for running a computation.
 */
class Program {
 public:
  /**
   * Constructor.
   * @param scope The scope containing all the runtime variables.
   * @param instrs The instructions belonging to this program.
   */
  Program(const std::shared_ptr<Scope>& scope, std::vector<std::unique_ptr<Instruction>>&& instrs)
      : scope_(scope), instrs_(std::move(instrs)) {}

  /**
   * Execute the program -- that is running all the instructions inside it.
   */
  void Execute() {
    for (auto& ins : instrs_) {
      ins->Run();
    }
  }

  /**
   * Get the number of instructions.
   */
  size_t size() const { return instrs_.size(); }

  ~Program() {}

 private:
  // We need to hold scope to assure tensors alive used in instructions.
  std::shared_ptr<Scope> scope_;
  std::vector<std::unique_ptr<Instruction>> instrs_;
};

/**
 * GraphCompiler compiles a graph and generate the runtime Program.
 */
class GraphCompiler final {
 public:
  GraphCompiler(Target target, const std::shared_ptr<Scope>& scope, const std::shared_ptr<Graph>& graph)
      : target_(std::move(target)), scope_(scope), graph_(graph), m_builder_(UniqName("module"), target) {}

  std::unique_ptr<Program> Build();

  void PrintFunc();

 private:
  ir::LoweredFunc GetOpFunc(const Node* node);

  std::string GenOpFuncName(const Node* node) const { return "fn_" + node->id(); }

  // TODO(haozech) add implementation
  std::vector<std::string> OpGetInputNames(const Node* node) const;
  // TODO(haozech) add implementation
  std::vector<std::string> OpGetOutputNames(const Node* node) const;

  std::vector<std::unique_ptr<Instruction>> BuildInstructions();

 private:
  Target target_;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Scope> scope_;

  std::unique_ptr<backends::Compiler> compiler_;

  lang::Module::Builder m_builder_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);
};

std::shared_ptr<Scope> BuildScope(Target target, const std::shared_ptr<Graph>& graph);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

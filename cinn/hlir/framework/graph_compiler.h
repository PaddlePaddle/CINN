#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/compiler.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"
#include "cinn/utils/timer.h"

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
      auto in_args  = ins->GetInArgs();
      auto out_args = ins->GetOutArgs();
      VLOG(3) << "Op in args: ";
      for (auto& in : in_args) {
        VLOG(3) << in << " ";
      }
      VLOG(3) << "Op out args: ";
      for (auto& out : out_args) {
        VLOG(3) << out << " ";
      }
      ins->Run();
#ifdef CINN_WITH_CUDA
      CUDA_CALL(cudaDeviceSynchronize());
#endif
    }
  }

  void ExecuteTest(int repeat_) {
    cinn::utils::Timer timer1;
    for (int i = 0; i < 100; i++) {
      for (auto& ins : instrs_) {
        ins->Run();
      }
    }
    timer1.Start();
    for (int i = 0; i < repeat_; i++) {
      for (auto& ins : instrs_) {
        ins->Run();
      }
    }
#ifdef CINN_WITH_CUDA
    CUDA_CALL(cudaDeviceSynchronize());
#endif
    double test_op_time = timer1.Stop() / repeat_;
    LOG(INFO) << "Repeat times: [" << repeat_ << "], average op time: [" << test_op_time << "] ms";
  }
  /**
   * Get the number of instructions.
   */
  size_t size() const { return instrs_.size(); }

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

  std::unique_ptr<Program> Build(const std::string& code = "");

  std::string GenSourceCode();

  void PrintFunc();

  const std::shared_ptr<Scope>& GetScope() const { return scope_; }

 private:
  ir::LoweredFunc GetOpFunc(const std::vector<Node*>& nodes);

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

  ir::Module::Builder m_builder_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);
};

std::shared_ptr<Scope> BuildScope(Target target,
                                  const std::shared_ptr<Graph>& graph,
                                  std::shared_ptr<Scope> scope = nullptr);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

#pragma once
#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/buffer_assign.h"
#include "cinn/optim/compute_inline_expand.h"
#include "cinn/optim/fold_cinn_call_arguments.h"
#include "cinn/optim/optimize.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/replace_call_with_expr.h"
#include "cinn/optim/tensor_write_tell.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/optim/transform_polyfor_to_for.h"
#include "cinn/poly/ast_gen.h"

namespace cinn {

namespace poly {
class Stage;
}  // namespace poly

namespace lang {
namespace detail {

/**
 * Mark the PolyFor as Vectorized if it is scheduled Vectorize in Stage.
 */
struct MarkVectorizeMutator : public ir::IRMutator<Expr*> {
  const std::map<std::string, ir::VectorizeInfo>& vectorizes;

  explicit MarkVectorizeMutator(const std::map<std::string /*tensor name*/, ir::VectorizeInfo>& vectorizes)
      : vectorizes(vectorizes) {}

  void operator()(Expr* expr) { ir::IRMutator<Expr*>::Visit(expr, expr); }

  // NOTE This mutator takes PolyFor as input, not For.
  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    stack.push_back(node);
    ir::IRMutator<ir::Expr*>::Visit(op, expr);
    stack.pop_back();
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    CHECK(tensor_n);
    auto it = vectorizes.find(tensor_n->name);
    if (it != vectorizes.end()) {
      stack[it->second.level]->set_vectorize_info(it->second);
      CHECK(it->second.valid());
    }
  }

  std::vector<ir::PolyFor*> stack;
};

/**
 * Mark the PolyFor as Unroll if is called Unroll in Stage.
 */
struct MarkUnrollMutator : public ir::IRMutator<Expr*> {
  std::map<std::string, std::set<int> /*level*/> unrolls;

  explicit MarkUnrollMutator(const std::map<std::string, std::set<int>>& unrolls) : unrolls(unrolls) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    stack.push_back(node);
    ir::IRMutator<>::Visit(op, expr);
    stack.pop_back();
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    CHECK(tensor_n);
    auto it = unrolls.find(tensor_n->name);
    if (it != unrolls.end()) {
      for (int level : it->second) {
        VLOG(1) << "Mark " << level << " Unrolled";
        CHECK_LT(level, stack.size());
        stack[level]->set_unrolled();
      }
    }
  }

  std::vector<ir::PolyFor*> stack;
};

/**
 * After the AstGen build the forloop from isl exprs, all the ISL Call nodes should be mapped to the corresponding CINN
 * expressions, there should be no remaining.
 */
void CheckNoIslCallRemains(const Expr* expr);

/**
 * \brief Lower a single group of nodes.
 *
 * We partition the whole computation of a function into several groups, each group is a basic element for ISL
 * polyhedral computation, that is, we transform a group into a isl domain and schedule, and generate ast latter.
 *
 * @param group A single schedule group containing several Stages and the scheduling order.
 * @param tuple_to_expr A map from isl set tuple name to CINN expressions.
 */
Expr LowerGroup(const poly::ScheduleGroup& group, const std::map<std::string, Expr>& tuple_to_expr);

/**
 * A Computation graph node.
 */
struct CompuGraphNode : public common::GraphNode {
  explicit CompuGraphNode(ir::Tensor tensor) : tensor(tensor) {}

  ir::Tensor tensor;

  std::string id() const override;
  const char* type_info() const override;
  static const char* __type_info__;
};

/**
 * \brief Create a computation graph using a tensor set.
 * It will deduce the temporary tensors not in the \p tensors.
 *
 * @param tensors the input/output tensors of a computation.
 * @param hide_inline hide inline tensor nodes.
 * @return a graph.
 *
 */
std::unique_ptr<common::Graph> CreateCompGraph(const std::vector<ir::Tensor>& tensors, bool hide_inline = false);

/**
 * \brief The implementation of Lower, transform the computation into a CINN function.
 */
struct LowerImpl {
  /**
   * \brief construct a LowerImpl.
   *
   * @param name The name of the generated LoweredFunc
   * @param tensor_args The argument list, with both the related placeholders and outputs
   * @param scalar_args The scalar arguments
   * @param temp_tensors
   * @return A CINN LoweredFunc
   *
   * A function is consist of two parts of information, one is the argument list which contains the input and
   * output, the other is a computation which read the inputs and write result to the outputs.
   *
   * The computation can be think as a SSA graph, the inputs and outputs of the graph should be contained in the union
   * set of \p tensor_args and \p scalar_args, but not all the variables used in the computation should in the union
   * set, those missing should be marked as temporary variables first.
   */
  LowerImpl(const std::string& name,
            const std::vector<Tensor>& tensor_args,
            const std::vector<Var>& scalar_args,
            const std::vector<Tensor>& temp_tensors);

  ir::LoweredFunc operator()();

  /**
   * \brief Generate the argument list of the final function.
   *
   * The argument list contains both the buffers(such as `cinn_buffer_t* X`) and necessary scalars (such as `int
   * batch_size`), all the scalar arguments will be in front of the tensor arguments in the function's argument list.
   *
   * @param func_body The body expression of the function.
   */
  inline std::vector<ir::Argument> GenerateFunctionArgumentList(Expr func_body);

  inline std::vector<ir::Buffer> CollectTemporaryBuffers();

  inline Expr GenerateFunctionBody(const poly::Schedule* schedule);

  inline void CheckAllTensorUsageInComputationContainsInArgs(poly::DataFlowGraph* graph);

  inline void CheckArgsUnique();

  //! All the tensor args including input, output and temporary tensors.
  inline std::vector<Tensor> all_tensor_args();

  /**
   * Collect the extra IO/control dependencies between stages.
   * @param stages the stages
   * @return the dependencies.
   */
  inline std::vector<std::pair<std::string, std::string>> CollectExtraDependencies(
      const std::vector<poly::Stage*>& stages);

  inline void InitStages() { stages_ = poly::GatherStagesInTensors(all_tensor_args()); }

  inline void InitTensorDic() {
    for (auto& tensor : all_tensor_args()) tensor_dic_.emplace(tensor->name, tensor);
  }

  inline void InitStageDic() {
    for (auto& stage : stages_) stage_dic_.emplace(stage->id(), stage);
  }

  /**
   * Safely get a tensor by \p name, check error if not found.
   * @param name the name of the target tensor.
   * @return the found tensor.
   */
  inline Tensor& TensorDicGet(const std::string& name) {
    auto it = tensor_dic_.find(name);
    CHECK(it != tensor_dic_.end()) << "Tensor [" << name << "] not found";
    return it->second;
  }

 private:
  std::string_view name_;
  const std::vector<Tensor>& tensor_args_;
  const std::vector<Var>& scalar_args_;
  const std::vector<Tensor>& temp_tensors_;

  std::vector<poly::Stage*> stages_;

  std::map<std::string, poly::Stage*> stage_dic_;
  std::map<std::string, Tensor> tensor_dic_;
};

/**
 * \brief Tell whether a tensor contains some GPU related information, such some schedule.
 */
bool TensorContainsGPUInfo(ir::Tensor t);

}  // namespace detail
}  // namespace lang
}  // namespace cinn

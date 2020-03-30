#include "cinn/lang/lower.h"

#include <map>
#include <set>
#include <stack>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/optimize.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/replace_call_with_expr.h"
#include "cinn/optim/tensor_write_tell.h"
#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace lang {

using ir::Tensor;
using poly::Stage;

/**
 * Mark the PolyFor as Vectorized if it is called Vectorize in Stage.
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

//! Lower a single group. A LoweredFunc is composed of several group.
Expr LowerGroup(const poly::ScheduleGroup& group, const std::map<std::string, Expr>& tuple_to_expr) {
  std::vector<poly::Stage*> stages;
  for (auto& node : group.nodes) {
    stages.push_back(node->stage);
  }

  // get isl generated expression
  isl::set context(Context::Global().isl_ctx(), "{:}");
  poly::AstGen gen(context, stages, group);
  isl::ast_node ast = gen.Build();
  ir::Expr e;
  poly::IslAstNodeToCinnExpr(ast, &e);

  // replace call to the corresponding statement
  for (auto& statement : tuple_to_expr) {
    auto axis_ast_map         = gen.axis2ast(statement.first);
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    std::map<std::string, Expr> axis;
    for (auto& item : axis_ast_map) {
      poly::IslAstExprToCinnExpr(item.second, &axis[item.first]);
    }
    VLOG(3) << "replacing " << statement.first << " to " << statement_candi_expr;
    optim::ReplaceCallWithExpr(&e, statement.first, statement_candi_expr, axis);
  }

  // mark vectorize.
  {
    std::map<std::string, ir::VectorizeInfo> vectorizes;
    for (auto& node : group.nodes) {
      if (node->stage->vectorize_info().valid()) {
        vectorizes[node->stage->id()] = node->stage->vectorize_info();
      }
    }
    MarkVectorizeMutator mutator(vectorizes);
    mutator(&e);
  }

  // mark unroll.
  {
    std::map<std::string, std::set<int>> unrolls;
    for (auto& node : group.nodes) {
      if (!node->stage->unroll_info().empty()) {
        unrolls[node->stage->id()] = node->stage->unroll_info();
      }
    }
    MarkUnrollMutator mutator(unrolls);
    mutator(&e);
  }

  return e;
}

//! Prepare the arguments of LoweredFunc.
std::vector<ir::Argument> PrepareArguments(const std::vector<Tensor>& tensors, const std::vector<Expr>& func_body) {
  std::vector<ir::Argument> args;
  optim::TensorWriteTeller teller;
  for (auto& expr : func_body) teller.Collect(&expr);

  std::set<std::string> arg_names;
  for (auto& tensor : tensors) {
    auto* tensor_node = tensor.As<ir::_Tensor_>();
    CHECK(!tensor_node->inlined());
    bool is_output = teller.IsWrite(tensor->name);

    // avoid duplicate
    if (arg_names.count(tensor_node->buffer->name)) continue;
    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(3) << "Collect " << (is_output ? "W" : "R") << " argument " << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }
  return args;
}

//! Lower the stages and get a LoweredFunc.
ir::LoweredFunc Lower(const std::string& name, const std::vector<Tensor>& args) {
  // make sure the graph's start-points in the args.

  auto stages             = poly::GatherStagesInTensors(args);
  auto extra_dependencies = poly::ExtractExtraDependencyFromStages(stages);
  auto graph              = poly::CreateGraph(stages, extra_dependencies);

  // Create a dic for stages and tensors.
  std::map<std::string, Stage*> stage_dic;
  std::map<std::string, Tensor> tensor_dic;
  for (auto& tensor : args) tensor_dic.emplace(tensor->name, tensor);
  for (auto& stage : stages) stage_dic.emplace(stage->id(), stage);
  // The placeholder Tensors are ignored in stages.
  CHECK_GE(tensor_dic.size(), stage_dic.size());
  CHECK_GE(args.size(), stage_dic.size()) << "tensor should duplicate name";

  std::set<std::string> args_names;
  for (auto& arg : args) {
    args_names.insert(arg->name);
  }
  CHECK_EQ(args.size(), args_names.size()) << "Tensor should have unique name";

  // collect the graph nodes of `args`
  std::vector<common::GraphNode*> input_graph_nodes;
  for (auto& node : graph->nodes()) {
    if (args_names.count(node->id())) {
      input_graph_nodes.push_back(node);
    }
  }

  auto depend_node_set = graph->dependencies(input_graph_nodes);
  // collect start points in the depend_node_set
  for (auto& node : depend_node_set) {
    CHECK(args_names.count(node->id())) << "The dependency tensor [" << node->id() << "] not in the inputs";
  }

  auto schedule = poly::CreateSchedule(stages, poly::ScheduleKind::Poly);

  // generate the expressions for each group.
  std::vector<Expr> exprs;
  CHECK_GT(schedule->groups.size(), 0) << "no group is generated";
  for (auto& group : schedule->groups) {
    CHECK_GT(group.nodes.size(), 0) << "group is empty";
    std::map<std::string, Expr> tuple_to_expr;
    for (auto& node : group.nodes) {
      auto& tensor = tensor_dic.at(node->id());
      // NOTE here just schedule the compute node.
      if (!tensor->is_compute_node()) continue;

      tuple_to_expr[tensor->name] = tensor->tensor_store_expanded_body();
    }

    Expr group_expr = LowerGroup(group, tuple_to_expr);
    VLOG(3) << "group expr:\n" << group_expr;
    exprs.push_back(group_expr);
  }

  Expr block = ir::Block::Make(exprs);

  // prepare arguments
  std::vector<ir::Argument> arguments = PrepareArguments(args, {block});

  auto func = ir::_LoweredFunc_::Make(name, arguments, block);
  auto res  = optim::Optimize(func);
  return ir::LoweredFunc(res.get());
}

}  // namespace lang
}  // namespace cinn

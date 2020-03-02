#include "cinn/lang/lower.h"

#include <cinn/ir/ir_mutator.h>

#include <map>
#include <set>

#include "cinn/ir/ir_printer.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/replace_call_with_expr.h"
#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace lang {

using ir::Tensor;
using poly::Stage;

Expr LowerGroup(const poly::detail::Group& group, const std::map<std::string, Expr>& tuple_to_expr) {
  std::vector<poly::Stage*> stages;
  for (auto& node : group.nodes) {
    stages.push_back(node->stage.get());
  }

  poly::PolyScheduler scheduler(stages);
  // TODO Schedule it.
  scheduler.BuildSchedule();

  isl::set context(Context::Global().isl_ctx(), "{:}");
  poly::AstGen gen(context, stages, scheduler);
  isl::ast_node ast = gen.Build();

  // set iterator names.

  ir::Expr e;
  poly::IslAstNodeToCinnExpr(ast, &e);

  for (auto& statement : tuple_to_expr) {
    auto axis_ast_map         = gen.axis2ast(statement.first);
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    std::map<std::string, Expr> axis;
    for (auto& item : axis_ast_map) {
      poly::IslAstExprToCinnExpr(item.second, &axis[item.first]);
    }
    optim::ReplaceCallWithExpr(&e, statement.first, statement_candi_expr, axis);
  }

  return e;
}

namespace {

struct WriteTeller : public ir::IRMutator<const Expr*> {
  std::set<std::string> buffer_written;

  void Visit(const Expr* expr, const Expr* op) override { IRMutator::Visit(expr, op); }

  void Visit(const ir::Load* expr, const Expr* op) override {
    buffer_written.insert(expr->buffer_var->name);
    IRMutator::Visit(expr, op);
  }
};

}  // namespace

std::vector<ir::Argument> PrepareArguments(const std::vector<Tensor>& tensors, const std::vector<Expr>& func_body) {
  std::vector<ir::Argument> args;
  WriteTeller teller;
  for (auto& expr : func_body) teller.Visit(&expr, &expr);

  for (auto& tensor : tensors) {
    bool is_input = teller.buffer_written.count(tensor->name);
    args.emplace_back(tensor->name,
                      ir::Argument::Kind::kBuffer,
                      tensor->type().ElementOf(),
                      tensor->shape.size(),
                      is_input ? ir::Argument::IO::kInput : ir::Argument::IO::kOutput);
  }
  return args;
}

std::vector<ir::LoweredFunc> Lower(const std::string& name, const std::vector<Tensor>& args) {
  // make sure the graph's start-points in the args.

  auto stages = poly::GatherStagesInTensors(args);
  auto graph  = poly::CreateGraph(stages);

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

  auto schedule = poly::CreateSchedule(stages);

  // generate the expressions for each group.
  std::vector<Expr> exprs;
  CHECK_GT(schedule->gened_groups().size(), 0) << "no group is generated";
  for (auto& group : schedule->gened_groups()) {
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
  // call passes
  optim::RemoveNestedBlock(&block);

  // prepare arguments
  std::vector<ir::Argument> arguments = PrepareArguments(args, {block});
  return {ir::_LoweredFunc_::Make(name, arguments, block)};
}

}  // namespace lang
}  // namespace cinn

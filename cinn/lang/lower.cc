#include "cinn/lang/lower.h"

#include <map>
#include <set>
#include <stack>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/replace_call_with_expr.h"
#include "cinn/optim/tensor_write_tell.h"
#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace lang {

using ir::Tensor;
using poly::Stage;

struct MarkVectorizeMutator : public ir::IRMutator<Expr*> {
  const std::map<std::string, ir::VectorizeInfo>& vectorizes;

  MarkVectorizeMutator(const std::map<std::string /*tensor name*/, ir::VectorizeInfo>& vectorizes)
      : vectorizes(vectorizes) {}

  void operator()(Expr* expr) { ir::IRMutator<Expr*>::Visit(expr, expr); }

  // NOTE This mutator takes PolyFor as input, not For.
  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node   = expr->As<ir::PolyFor>();
    last_polyfor = node;
    ir::IRMutator<ir::Expr*>::Visit(op, expr);
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    CHECK(tensor_n);
    auto it = vectorizes.find(tensor_n->name);
    if (it != vectorizes.end()) {
      CHECK(last_polyfor);
      last_polyfor->vectorize_info = it->second;
    }
  }

  ir::PolyFor* last_polyfor{};
};

/**
 * Expand the split transforms.
 * This should takes the expression generated from isl ast as input(without relacing the statement with the real
 * computation), it takes each Call to identify the statement. Each time it can only deal with one statement.
 */
struct SplitExpandMutator : public ir::IRMutator<Expr*> {
  SplitExpandMutator(const std::string& statement, const std::map<std::string, poly::SplitRestStrategy>& strategies)
      : statement_(statement), strategies_(strategies) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    forloop_stack_.push(expr);

    ir::IRMutator<>::Visit(op, expr);

    forloop_stack_.pop();

    // The Split transform always split one forloop into outer and inner, and we do separation on the inner one, so
    // there should be at least one forloop remaining if the current level is the inner.
    if (!forloop_stack_.empty()) {
      if (cur_statement_ == statement_ && strategies_.count(op->iterator->name) &&
          strategies_.at(op->iterator->name) == poly::SplitRestStrategy::kSeparate) {
        auto* outer = forloop_stack_.top()->As<ir::PolyFor>();
        DoSaparation(outer, expr);
      }
    }
  }

  void Visit(const ir::Call* op, Expr* expr) override {
    auto* node = expr->As<ir::Call>();
    // We reach the call node that represents the statment, just mark the current(innermost) forloop to separate.
    if (node->call_type == ir::Call::CallType::ISL) {
      cur_statement_ = node->name;
    }
  }

  //! Do the separation on the \p inner forloop, add new forloops to \p outer forloop.
  void DoSaparation(ir::PolyFor* outer, Expr* inner) {
    VLOG(3) << "Doing separation";
    auto* inner_node = inner->As<ir::PolyFor>();
    CHECK(inner_node);
    auto* lt = inner_node->condition.As<ir::LT>();
    auto* le = inner_node->condition.As<ir::LE>();

    auto create_forloop = [&](Expr cond) {
      return ir::PolyFor::Make(inner_node->iterator,
                               inner_node->init,
                               cond,
                               inner_node->inc,
                               inner_node->for_type,
                               inner_node->device_api,
                               inner_node->body);
    };

    auto insert_new_forloops_to_upper = [&](ir::PolyFor* origin, Expr if_then_else) {
      auto* outer_block = outer->body.As<ir::Block>();
      CHECK(outer_block);
      auto it = std::find_if(outer_block->stmts.begin(), outer_block->stmts.end(), [&](const Expr& e) {
        auto* a_for = e.As<ir::PolyFor>();
        if (!a_for) return false;
        return a_for == origin;
      });
      CHECK(it != outer_block->stmts.end());

      *it = if_then_else;
    };

    Expr cond0, cond1;
    if (!(lt || le)) {
      LOG(ERROR) << "The condition of the forloop don't contains LT or LE operator, skip seperation, the condition is "
                 << inner_node->condition;
      return;
    }

    ir::Min* min_n = lt ? lt->b.As<ir::Min>() : le->b.As<ir::Min>();

    if (min_n) {
      auto upper_bound0 = min_n->a;
      auto upper_bound1 = min_n->b;

      Expr forloop0, forloop1;
      if (lt) {
        forloop0 = create_forloop(ir::LT::Make(Expr(inner_node->iterator), upper_bound0));
        forloop1 = create_forloop(ir::LT::Make(Expr(inner_node->iterator), upper_bound1));
      } else {
        forloop0 = create_forloop(ir::LE::Make(Expr(inner_node->iterator), upper_bound0));
        forloop1 = create_forloop(ir::LE::Make(Expr(inner_node->iterator), upper_bound1));
      }

      // the new forloops should be wrapped by a if-then-else
      Expr if_then_else_cond = ir::LE::Make(upper_bound0, upper_bound1);
      auto if_then_else      = ir::IfThenElse::Make(if_then_else_cond, forloop0, forloop1);
      VLOG(2) << "Separate two new forloops";
      VLOG(2) << forloop0;
      VLOG(2) << forloop1;
      insert_new_forloops_to_upper(inner_node, if_then_else);
    }
  }

 private:
  std::string statement_;
  //! A stack to record the forloops call stack to the current statement.
  std::stack<ir::Expr*> forloop_stack_;
  const std::map<std::string, poly::SplitRestStrategy>& strategies_;
  ir::Expr* forloop_to_separate_{};
  //! The statement in the innermost forloop, used to determine whether the forloops in the stack need to separate.
  std::string cur_statement_;
};  // namespace lang

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

  for (auto& stage : stages) {
    VLOG(3) << "run Split separation on " << stage->id() << " " << stage->split_strageties().size() << " strategies";
    SplitExpandMutator(stage->id(), stage->split_strageties())(&e);
  }

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
  std::map<std::string, ir::VectorizeInfo> vectorizes;
  for (auto& node : group.nodes) {
    if (node->stage->vectorize_info().valid()) {
      vectorizes[node->stage->id()] = node->stage->vectorize_info();
    }
  }
  MarkVectorizeMutator mutator(vectorizes);
  mutator(&e);

  return e;
}

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

std::vector<ir::LoweredFunc> Lower(const std::string& name, const std::vector<Tensor>& args) {
  // make sure the graph's start-points in the args.

  auto stages = poly::GatherStagesInTensors(args);
  auto graph  = poly::CreateGraph(stages);
  LOG(INFO) << "Graph:\n" << graph->Visualize();

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
      LOG(INFO) << "graph node " << node->id();
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

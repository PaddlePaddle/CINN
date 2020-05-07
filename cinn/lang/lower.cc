#include "cinn/lang/lower.h"

#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include <utility>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/fold_call_arguments.h"
#include "cinn/optim/optimize.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/replace_call_with_expr.h"
#include "cinn/optim/tensor_write_tell.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/optim/transform_polyfor_to_for.h"
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

//! Mark the PolyFor as Unroll if is called Unroll in Stage.
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

void CheckNoIslCallRemains(const Expr* expr) {
  auto isl_calls = ir::CollectIRNodes(*expr, [](const Expr* expr) {
    return expr->As<ir::Call>() && expr->As<ir::Call>()->call_type == ir::Call::CallType ::ISL;
  });
#ifdef CINN_DEBUG
  for (auto& item : isl_calls) {
    LOG(ERROR) << "ISL call: " << item;
  }
#endif
  CHECK(isl_calls.empty()) << "Some ISL call nodes remained";
}

/**
 * Lower a single group of nodes.
 *
 * It first union all the domains of the stages into a UnionSet, and all the transforms into a UnionMap.
 *
 */
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

  VLOG(3) << "ast to expr: \n" << e << std::endl;

  // replace call to the corresponding statement
  for (auto& statement : tuple_to_expr) {
    if (!gen.ContainsStatement(statement.first)) continue;
    auto axis_ast_map         = gen.axis2ast(statement.first);
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    std::map<std::string, Expr> axis;
    for (auto& item : axis_ast_map) {
      poly::IslAstExprToCinnExpr(item.second, &axis[item.first]);
    }
    VLOG(3) << "replacing " << statement.first << " to " << statement_candi_expr;
    optim::ReplaceCallWithExpr(&e, statement.first, statement_candi_expr, axis);
  }
  CheckNoIslCallRemains(&e);

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

  // mark gpu
  {
    optim::forloop_infos_t forloop_infos;
    for (auto* stage : stages) {
      forloop_infos[stage->id()] = stage->forloop_infos();
    }
    optim::TransformGpuForloop(forloop_infos, &e);
  }

  return e;
}

struct LowerImpl {
  LowerImpl(const std::string& name,
            const std::vector<Tensor>& tensor_args,
            const std::vector<Var>& scalar_args,
            const std::vector<Tensor>& temp_tensors)
      : name_(name), tensor_args_(tensor_args), scalar_args_(scalar_args), temp_tensors_(temp_tensors) {
    InitStages();
    InitStageDic();
    InitTensorDic();
    CheckArgsUnique();
  }

  ir::LoweredFunc operator()() {
    CHECK(!stages_.empty()) << "At least one stage is needed";

    auto deps     = collect_extra_dependencis(stages_);
    auto graph    = poly::CreateGraph(stages_, deps);
    auto schedule = poly::CreateSchedule(stages_, poly::ScheduleKind::Poly, deps);

    CheckAllTensorUsageInComputationContainsInArgs(graph.get());

    auto func_body      = GenFnBody(schedule.get());
    auto func_args      = GenFnArgs(func_body);
    auto func_temp_bufs = GenTempBuffers();

    auto func = ir::_LoweredFunc_::Make(std::string(name_), func_args, func_body, func_temp_bufs);
    auto res  = optim::Optimize(func, FLAGS_cinn_runtime_display_debug_info);
    return ir::LoweredFunc(res.get());
  }

  inline std::vector<ir::Argument> GenFnArgs(Expr func_body) {
    std::vector<ir::Argument> args;
    optim::TensorWriteTeller teller;
    teller.Collect(&func_body);

    std::set<std::string> arg_names;

    for (auto& scalar : scalar_args_) {
      if (arg_names.count(scalar->name)) continue;
      auto* scalar_node = scalar.As<ir::_Var_>();
      CHECK(scalar_node->type().valid());
      arg_names.insert(scalar->name);

      args.emplace_back(scalar, ir::Argument::IO::kInput);
    }

    for (auto& tensor : tensor_args_) {
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

  inline std::vector<ir::Buffer> GenTempBuffers() {
    std::vector<ir::Buffer> res;
    for (auto& x : temp_tensors_) {
      CHECK(!x->inlined());
      res.push_back(x->buffer);
    }
    return res;
  }

  inline Expr GenFnBody(const poly::Schedule* schedule) {
    // generate the expressions for each group.
    std::vector<Expr> exprs;
    std::map<std::string, Expr> tuple_to_expr;
    CHECK_GT(schedule->groups.size(), 0) << "no group is generated";
    for (auto& group : schedule->groups) {
      CHECK_GT(group.nodes.size(), 0) << "group is empty";
      for (auto& node : group.nodes) {
        auto& tensor = TensorDicGet(node->id());
        // NOTE here just schedule the compute node.
        if (!tensor->is_compute_node() && !tensor->is_call_node()) continue;

        tuple_to_expr[tensor->name] = tensor->tensor_store_expanded_body();
      }

      Expr group_expr = LowerGroup(group, tuple_to_expr);
      VLOG(3) << "group expr:\n" << group_expr;
      exprs.push_back(group_expr);
    }

    Expr body = ir::Block::Make(exprs);
    return body;
  }

  inline void CheckAllTensorUsageInComputationContainsInArgs(poly::DataFlowGraph* graph) {
    std::set<std::string> tensor_arg_names;
    for (auto& tensor : all_tensor_args()) {
      tensor_arg_names.insert(tensor->name);
    }

    std::vector<common::GraphNode*> tensor_arg_nodes;
    for (auto& node : graph->nodes()) {
      if (tensor_arg_names.count(node->id())) {
        tensor_arg_nodes.push_back(node);
      }
    }

    auto tensor_arg_depend_node_set = graph->dependencies(tensor_arg_nodes);

    // check the tensor arguments(including the outputs)'s dependencies are also locate in the arguments

    for (auto& node : tensor_arg_depend_node_set) {
      CHECK(tensor_arg_names.count(node->id()))
          << "Found the dependency tensor [" << node->id() << "] not in the arguments";
    }
  }

  inline void CheckArgsUnique() {
    auto _all_tensor_args = all_tensor_args();
    std::unordered_set<std::string> arg_names;
    for (auto& tensor : _all_tensor_args) {
      CHECK(!arg_names.count(tensor->name))
          << "The argument of the function, tensor [" << tensor->name << "] duplicates";
      arg_names.insert(tensor->name);
    }

    for (auto& scalar : scalar_args_) {
      CHECK(!arg_names.count(scalar->name))
          << "The argument of the function, scalar [" << scalar->name << "] duplicates";
      arg_names.insert(scalar->name);
    }
  }

  //! All the tensor args including input, output and temporary tensors.
  inline std::vector<Tensor> all_tensor_args() {
    std::vector<Tensor> res = tensor_args_;
    res.insert(res.end(), temp_tensors_.begin(), temp_tensors_.end());
    return res;
  }

  //! Get all the extra dependencies.
  inline std::vector<std::pair<std::string, std::string>> collect_extra_dependencis(const std::vector<Stage*>& stages) {
    auto call_dependencies  = poly::ExtractLinksFromCalls(tensor_args_, false);
    auto extra_dependencies = poly::ExtractExtraDepLinksFromStages(stages);  // deal with the `compute_at` dependencies.

    extra_dependencies.insert(std::end(extra_dependencies), call_dependencies.begin(), call_dependencies.end());
    return extra_dependencies;
  }

  inline void InitStages() { stages_ = poly::GatherStagesInTensors(all_tensor_args()); }

  inline void InitTensorDic() {
    for (auto& tensor : all_tensor_args()) tensor_dic_.emplace(tensor->name, tensor);
  }

  inline void InitStageDic() {
    for (auto& stage : stages_) stage_dic_.emplace(stage->id(), stage);
  }

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

  std::vector<Stage*> stages_;

  std::map<std::string, Stage*> stage_dic_;
  std::map<std::string, Tensor> tensor_dic_;
};

ir::LoweredFunc Lower(const std::string& name,
                      const std::vector<Tensor>& tensor_args,
                      const std::vector<Var>& scalar_args,
                      const std::vector<Tensor>& temp_tensors,
                      Module::Builder* b) {
  if (!temp_tensors.empty()) {
    CHECK(b) << "Module should be set to hold the temporary buffers";

    for (auto& temp_tensor : temp_tensors) {
      CHECK(!temp_tensor->inlined()) << "The tensor arguments of function should bind to buffers";
      b->AddBuffer(temp_tensor->buffer);
    }
  }

  auto res = LowerImpl(name, tensor_args, scalar_args, temp_tensors)();

  if (b) {
    b->AddFunction(res);
  }
  return res;
}

}  // namespace lang
}  // namespace cinn

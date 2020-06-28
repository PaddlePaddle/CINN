#include "cinn/lang/lower_impl.h"

#include <queue>
#include <unordered_set>

namespace cinn {
namespace lang {
namespace detail {

void CheckNoIslCallRemains(Expr* expr) {
  auto isl_calls = ir::CollectIRNodes(
      *expr, [](const Expr* expr) { return expr->As<ir::Call>() && expr->As<ir::Call>()->is_isl_call(); });
#ifdef CINN_DEBUG
  for (auto& item : isl_calls) {
    LOG(ERROR) << "ISL call: " << item;
  }
#endif
  if (!isl_calls.empty()) {
    LOG(WARNING) << "Some ISL call nodes remained, get " << isl_calls.size() << " isl_calls, the first one is "
                 << *isl_calls.begin();
  }
}

Expr LowerGroup(const poly::ScheduleGroup& group, const std::map<std::string, Expr>& tuple_to_expr) {
  std::vector<poly::Stage*> stages;
  for (auto& node : group.nodes) {
    if (node->stage->has_expression()) {
      stages.push_back(node->stage);
      VLOG(1) << "stage expr " << node->stage->expr();
    } else {
      VLOG(1) << "stage expression is null: " << node->stage->domain();
    }
  }

  if (stages.empty()) return Expr();

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
#ifdef CINN_WITH_CUDA
  {
    optim::forloop_infos_t forloop_infos;
    for (auto* stage : stages) {
      forloop_infos[stage->id()] = stage->forloop_infos();
    }
    optim::TransformGpuForloop(forloop_infos, &e);
  }
#endif

  return e;
}

bool TensorContainsGPUInfo(ir::Tensor t) {
  if (t->inlined()) return false;
  if (t->stage()) {
    for (auto& info : t->stage()->forloop_infos()) {
      if (info.second.device == ir::DeviceAPI::GPU) {
        return true;
      }
    }
  }
  return false;
}

LowerImpl::LowerImpl(const std::string& name,
                     const std::vector<Tensor>& tensor_args,
                     const std::vector<Var>& scalar_args,
                     const std::vector<Tensor>& temp_tensors)
    : name_(name), tensor_args_(tensor_args), scalar_args_(scalar_args), temp_tensors_(temp_tensors) {
  // Inline expand
  for (auto& t : tensor_args_) {
    Expr tt(t);
    optim::ComputeInlineExpand(&tt);
  }
  for (auto& t : temp_tensors_) {
    Expr tt(t);
    optim::ComputeInlineExpand(&tt);
  }

  InitStages();
  InitStageDic();
  InitTensorDic();
  CheckArgsUnique();
}

ir::LoweredFunc LowerImpl::operator()() {
  CHECK(!stages_.empty()) << "At least one stage is needed";

  auto deps  = CollectExtraDependencies(stages_);
  auto graph = poly::CreateGraph(stages_, deps);
  LOG(INFO) << "graph:\n" << graph->Visualize();
  auto schedule = poly::CreateSchedule(stages_, poly::ScheduleKind::Poly, deps);

  CheckAllTensorUsageInComputationContainsInArgs(graph.get());

  auto func_body = GenerateFunctionBody(schedule.get());

  auto tensor_map = optim::InitialAssignBuffer(&func_body);
  // copy the tensor(with buffer assigned) back to func's args.
  {
    for (auto& arg : tensor_args_) {
      if (arg->is_placeholder_node()) continue;
      if (arg->buffer.defined()) continue;
      if (arg->body().As<ir::Call>() && arg->body().type().is_void()) continue;  // extern call
      Reference(&arg)->buffer = tensor_map.at(arg->name)->buffer;
    }
  }

  auto func_args      = GenerateFunctionArgumentList(func_body);
  auto func_temp_bufs = CollectTemporaryBuffers();

  auto func = ir::_LoweredFunc_::Make(std::string(name_), func_args, func_body, func_temp_bufs);

  // some necessary modification.
  optim::ComputeInlineExpand(&func->body);

  auto res = optim::Optimize(func, FLAGS_cinn_runtime_display_debug_info);
  return ir::LoweredFunc(res.get());
}

std::vector<ir::Argument> LowerImpl::GenerateFunctionArgumentList(Expr func_body) {
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
    bool is_output    = teller.IsWrite(tensor->name);
    VLOG(5) << "tensor argument " << tensor->name << " buffer " << tensor->buffer->name;

    // avoid duplicate
    if (!tensor_node->buffer.defined()) continue;
    // if a argument is already marked as kInput, mark it as kOutput and move it to the back.
    if (arg_names.count(tensor_node->buffer->name)) {
      auto it = std::find_if(
          args.begin(), args.end(), [&](const ir::Argument& x) { return x.name() == tensor_node->buffer->name; });
      CHECK(it != args.end());
      if (it->is_input()) {
        args.erase(it);
      } else if (it->is_output()) {
        continue;
      }
    }

    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(3) << "Collect " << (is_output ? "W" : "R") << " argument " << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }

  return args;
}

std::vector<ir::Buffer> LowerImpl::CollectTemporaryBuffers() {
  std::vector<ir::Buffer> res;
  for (auto& x : temp_tensors_) {
    CHECK(!x->inlined());
    res.push_back(x->buffer);
  }
  return res;
}

Expr LowerImpl::GenerateFunctionBody(const poly::Schedule* schedule) {
  // generate the expressions for each group.
  std::vector<Expr> exprs;
  std::map<std::string, Expr> tuple_to_expr;
  CHECK(!schedule->groups.empty()) << "no group is generated";
  for (auto& group : schedule->groups) {
    CHECK_GT(group.nodes.size(), 0) << "group is empty";
    for (auto& node : group.nodes) {
      if (!tensor_dic_.count(node->id())) continue;
      auto& tensor = TensorDicGet(node->id());
      if (!tensor->has_expression()) continue;
      tuple_to_expr[tensor->name] = tensor->tensor_store_expanded_body();
    }

    Expr group_expr = LowerGroup(group, tuple_to_expr);
    if (group_expr.defined()) {
      VLOG(1) << "group expr:\n" << group_expr;
      exprs.push_back(group_expr);
    }
  }

  Expr body = ir::Block::Make(exprs);
  return body;
}

void LowerImpl::CheckAllTensorUsageInComputationContainsInArgs(poly::DataFlowGraph* graph) {
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

void LowerImpl::CheckArgsUnique() {
  auto _all_tensor_args = all_tensor_args();
  std::unordered_set<std::string> arg_names;
  for (auto& tensor : _all_tensor_args) {
    CHECK(!arg_names.count(tensor->name)) << "The argument of the function, tensor [" << tensor->name << "] duplicates";
    arg_names.insert(tensor->name);
  }

  for (auto& scalar : scalar_args_) {
    CHECK(!arg_names.count(scalar->name)) << "The argument of the function, scalar [" << scalar->name << "] duplicates";
    arg_names.insert(scalar->name);
  }
}

std::vector<std::pair<std::string, std::string>> LowerImpl::CollectExtraDependencies(
    const std::vector<poly::Stage*>& stages) {
  auto call_dependencies  = poly::ExtractLinksFromCalls(tensor_args_, false);
  auto extra_dependencies = poly::ExtractExtraDepLinksFromStages(stages);  // deal with the `compute_at` dependencies.

  extra_dependencies.insert(std::end(extra_dependencies), call_dependencies.begin(), call_dependencies.end());
  return extra_dependencies;
}

std::vector<Tensor> LowerImpl::all_tensor_args() {
  std::vector<Tensor> res = tensor_args_;
  res.insert(res.end(), temp_tensors_.begin(), temp_tensors_.end());
  return res;
}

const char* CompuGraphNode::__type_info__ = "ComputeGraphNode";
const char* CompuGraphNode::type_info() const { return __type_info__; }
std::string CompuGraphNode::id() const {
  CHECK(tensor.defined());
  return tensor->name;
}

void CreateCompGraphHelper(common::Graph* graph, ir::Tensor& t, Expr e, bool hide_inline) {
  bool hide_t               = hide_inline && t->inlined();
  common::GraphNode* t_node = graph->RetriveNode(t->name);
  if (!t_node && !hide_t) {
    t_node = graph->RegisterNode(t->name, new CompuGraphNode(t));
  }

  auto e_tensor = e.as_tensor_ref();
  if (e_tensor.defined()) {
    auto* e_node = graph->RetriveNode(e_tensor->name);
    if (!e_node && !(hide_inline && e_tensor->inlined())) {
      e_node = graph->RegisterNode(e_tensor->name, new CompuGraphNode(e_tensor));
    }
    if (!hide_t && t_node && e_node) e_node->LinkTo(t_node);
  }

  for (auto* e_dep : e->expr_fields()) {
    if (e_tensor.defined()) {
      CreateCompGraphHelper(graph, e_tensor, *e_dep, hide_inline);
    } else {
      CreateCompGraphHelper(graph, t, *e_dep, hide_inline);
    }
  }
}

std::unique_ptr<common::Graph> CreateCompGraph(const std::vector<ir::Tensor>& tensors, bool hide_inline) {
  auto graph = std::make_unique<common::Graph>();

  for (auto& t : tensors) {
    auto tc = t;
    if (hide_inline && tc->inlined()) continue;
    for (auto& e : tc->expr_fields()) {
      CreateCompGraphHelper(graph.get(), tc, *e, hide_inline);
    }
  }

  return graph;
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn

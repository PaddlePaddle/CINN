#include "cinn/lang/lower_impl.h"

#include <algorithm>
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

  VLOG(1) << "ast to expr: \n" << e << std::endl;

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

const char* CompuGraphNode::__type_info__ = "ComputeGraphNode";
const char* CompuGraphNode::type_info() const { return __type_info__; }
std::string CompuGraphNode::id() const {
  CHECK(tensor.defined());
  return tensor->name;
}

void CreateCompGraphHelper(common::Graph* graph, ir::Tensor& t, Expr e, bool hide_inline) {  // NOLINT
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

  // consider the extra_depend field in tensor.stage
  for (auto* node : graph->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    for (auto& tname : cnode->tensor->stage()->extra_depend_stages()) {
      auto* depend_node = graph->RetriveNode(tname);
      if (depend_node) {
        depend_node->LinkTo(node);
      }
    }
  }

  return graph;
}

void CloneCompGraph(const common::Graph& comp_graph, poly::DataFlowGraph* dfg) { CHECK(dfg); }

void LowerImpl::CheckArgsUnique() {
  std::unordered_set<std::string> arg_names;
  for (auto& tensor : tensor_args_) {
    CHECK(!tensor->inlined()) << "Inline tensor cannot be argument of function";
    CHECK(!arg_names.count(tensor->name)) << "The argument of the function, tensor [" << tensor->name << "] duplicates";
    arg_names.insert(tensor->name);
    if (!tensor->buffer.defined()) {
      LOG(ERROR) << "tensor [" << tensor->name << "] buffer is null";
      continue;
    }
    arg_names.insert(tensor->buffer->name);
  }

  for (auto& scalar : scalar_args_) {
    CHECK(!arg_names.count(scalar->name)) << "The argument of the function, scalar [" << scalar->name << "] duplicates";
    arg_names.insert(scalar->name);
  }
}

std::vector<ir::Argument> LowerImpl::GenerateFunctionArgumentList(Expr fn_body) {
  CheckArgsUnique();

  std::vector<ir::Argument> args;
  optim::TensorWriteTeller teller;
  teller.Collect(&fn_body);

  std::set<std::string> arg_names;

  for (auto& scalar : scalar_args_) {
    CHECK(!arg_names.count(scalar->name));
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

std::vector<Tensor> LowerImpl::CollectTemporaryTensors() {
  // a temporary should be in the comp_graph but not contained in the tensor_args.
  std::unordered_map<std::string, Tensor> tensor_arg_map = GenTensorArgMap();
  std::unordered_map<std::string, Tensor> temp_tensor_map;

  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    if (!tensor_arg_map.count(cnode->tensor->name)) {
      temp_tensor_map[cnode->tensor->name] = cnode->tensor;
    }
  }

  std::vector<Tensor> temp_tensors;
  std::transform(temp_tensor_map.begin(),
                 temp_tensor_map.end(),
                 std::back_inserter(temp_tensors),
                 [&](const decltype(temp_tensor_map)::value_type& x) { return x.second; });
  return temp_tensors;
}

std::unordered_map<std::string, Tensor> LowerImpl::GenTensorArgMap() {
  std::unordered_map<std::string, Tensor> map;
  for (auto& t : tensor_args_) {
    map[t->name] = t;
  }
  return map;
}

std::unordered_map<std::string, Tensor> LowerImpl::GenAllTensorMap() {
  std::unordered_map<std::string, Tensor> map;
  for (auto& t : CollectAllTensors()) {
    map[t->name] = t;
  }
  return map;
}

ir::LoweredFunc LowerImpl::operator()() {
  // get tensors
  std::vector<Tensor> all_tensors;

  std::vector<poly::Stage*> stages = poly::GatherStagesInTensors(CollectAllTensors());

  auto deps     = CollectExtraDependencies();
  auto schedule = poly::CreateSchedule(
      stages, poly::ScheduleKind::Poly, std::vector<std::pair<std::string, std::string>>(deps.begin(), deps.end()));

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

  auto func_args         = GenerateFunctionArgumentList(func_body);
  auto func_temp_tensors = CollectTemporaryTensors();
  std::vector<ir::Buffer> temp_buffers;
  std::unordered_set<std::string> buffer_name_set;
  // TODO(Superjomn) write buffer latter.
  for (auto& t : func_temp_tensors) {
    if (t->buffer.defined() && !buffer_name_set.count(t->buffer->name)) {
      temp_buffers.push_back(t->buffer);
      buffer_name_set.insert(t->buffer->name);
    }
  }

  auto func = ir::_LoweredFunc_::Make(fn_name_, func_args, func_body, temp_buffers);

  // some necessary modification.
  optim::ComputeInlineExpand(&func->body);

  auto res = optim::Optimize(func, FLAGS_cinn_runtime_display_debug_info);
  return ir::LoweredFunc(res.get());
}

std::vector<Tensor> LowerImpl::CollectAllTensors() {
  std::vector<Tensor> tensors;
  auto [nodes, edges] = compu_graph_->topological_order();  // NOLINT
  for (auto* node : nodes) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    tensors.push_back(cnode->tensor);
  }
  return tensors;
}

std::set<std::pair<std::string, std::string>> LowerImpl::CollectExtraDependencies() const {
  std::set<std::pair<std::string, std::string>> deps;
  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    for (auto& dep : cnode->tensor->stage()->extra_depend_stages()) {
      deps.emplace(dep, cnode->tensor->name);
    }
  }
  return deps;
}

Expr LowerImpl::GenerateFunctionBody(const poly::Schedule* schedule) {
  // generate the expressions for each group.
  std::vector<Expr> exprs;
  auto tensor_map = GenAllTensorMap();
  std::map<std::string, Expr> tuple_to_expr;
  CHECK(!schedule->groups.empty()) << "no group is generated";
  for (auto& group : schedule->groups) {
    CHECK_GT(group.nodes.size(), 0) << "group is empty";
    for (auto& node : group.nodes) {
      if (!tensor_map.count(node->id())) continue;
      auto& tensor = tensor_map[node->id()];
      if (!tensor->has_expression()) continue;
      tuple_to_expr[tensor->name] = tensor->tensor_store_expanded_body();
    }

    Expr group_expr = LowerGroup(group, tuple_to_expr);
    if (group_expr.defined()) {
      VLOG(3) << "group expr:\n" << group_expr;
      exprs.push_back(group_expr);
    }
  }

  Expr body = ir::Block::Make(exprs);
  return body;
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn

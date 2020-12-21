#include "cinn/lang/lower_impl.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute_at_postprocess.h"
#include "cinn/optim/cache_read_write_replace.h"
#include "cinn/poly/stage.h"

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

Expr LowerGroup(const poly::ScheduleGroup& group,
                const std::map<std::string, Expr>& tuple_to_expr,
                std::map<std::string, ir::Tensor>* global_tensor_map,
                StageMap stage_map,
                ir::CudaAxisInfo* cuda_axis_info) {
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
  // now we get a workable expression, but the statement are something like `B(((16 * po0) + po1), po2)`, we need to
  // transform this to some realworld statement in CINN.

  VLOG(1) << "ast to expr: \n" << e << std::endl;

  // replace isl call to the corresponding CINN statement, we need to replace the axis at the same time.
  for (auto& statement : tuple_to_expr) {
    VLOG(2) << "LowerGroup working on statement: " << statement.first;
    if (!gen.ContainsStatement(statement.first)) continue;
    // the axis_ast_map contains the axis from the original (like `i`) to the transformed (like `i+3`).
    auto axis_expr_map = gen.axis2expr(statement.first);
    for (auto& item : axis_expr_map) {
      VLOG(4) << "statement ast map axis [" << item.first << "] to "
              << "[" << item.second << "]";
    }

    // the original CINN statements.
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    VLOG(3) << "replacing " << statement.first << " to " << statement_candi_expr;
    optim::ReplaceIslCallWithExpr(&e, statement.first, statement_candi_expr, axis_expr_map);
  }
  CheckNoIslCallRemains(&e);

  optim::CacheReadWriteReplace(&e, stage_map, global_tensor_map);

  // deal with the compute_at relations
  ProcessComputeAtInfo(&e, stage_map);

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

  // mark gpu threads
#ifdef CINN_WITH_CUDA
  {
    optim::forloop_infos_t forloop_infos;
    for (auto* stage : stages) {
      // transform the level identified for infors to iter name identified.
      auto iters = common::GatherItersToTensorProducer(stage->id(), &e);
      std::map<std::string, poly::StageForloopInfo> for_infos;
      for (auto& item : stage->forloop_infos()) {
        CHECK_LT(item.first, iters.size());
        for_infos[iters[item.first]] = item.second;
      }

      forloop_infos[stage->id()] = for_infos;
    }
    optim::TransformGpuForloops(forloop_infos, global_tensor_map, &e);
    auto axis_info = optim::GatherAxisInfoFromStages(stages);
    if (axis_info.valid()) cuda_axis_info->ExtendWith(axis_info);
  }
#endif  // CINN_WITH_CUDA

  return e;
}

bool TensorContainsGPUInfo(ir::Tensor t, poly::Stage* stage) {
  if (stage->inlined()) return false;
  if (stage) {
    for (auto& info : stage->forloop_infos()) {
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

/**
 * \brief Add nodes to graph with dependencies.
 * We create a computation graph based on the tensor dependency relations.
 * NOTE The graph will contain the inline tensors so that the dependency will be reserved.
 * @param graph The graph
 * @param t The tensor.
 * @param stages The stage map.
 */
void CreateCompGraphWithInlineTensors(common::Graph* graph,
                                      const ir::Tensor& t,
                                      StageMap stages,
                                      std::set<ir::Tensor>* visited) {
  if (visited->count(t)) return;
  common::GraphNode* t_node = graph->RetriveNode(t->name);
  if (!t_node) {
    t_node = graph->RegisterNode(t->name, new CompuGraphNode(t));
  }

  visited->insert(t);

  // collect dependency tensors of t
  // here we just collect the tensors in Load nodes
  // NOTE there may be some other cases.
  auto deps = ir::CollectLoadTensors(t->body(), [](const Expr* x) { return x->as_tensor(); });
  for (const auto& dep : deps) {
    auto e_tensor = dep.as_tensor_ref();
    auto* e_node  = graph->RetriveNode(e_tensor->name);
    if (!e_node) {
      e_node = graph->RegisterNode(e_tensor->name, new CompuGraphNode(e_tensor));
    }
    e_node->LinkTo(t_node);
    if (!visited->count(e_tensor)) {
      CreateCompGraphWithInlineTensors(graph, e_tensor, stages, visited);
    }
  }
}

std::unique_ptr<common::Graph> CreateCompGraphWithInlineTensorHidden(const std::vector<ir::Tensor>& tensors,
                                                                     StageMap stages) {
  // create a graph with inline tensor first.
  std::unique_ptr<common::Graph> graph(new common::Graph);
  std::set<ir::Tensor> visited;
  for (auto& t : tensors) {
    CreateCompGraphWithInlineTensors(graph.get(), t, stages, &visited);
  }

  // greedy remove the inline tensor, each time merge the inputs of an inline tensor to its sink node.

  std::set<common::GraphNode*> inline_nodes;
  do {
    inline_nodes = graph->CollectNodes([&](const common::GraphNode* x) {
      auto* comp_node = x->safe_as<CompuGraphNode>();
      return stages[comp_node->tensor]->inlined();
    });
    if (inline_nodes.empty()) break;

    /*
     * A -> inlined -> B
     * C /
     * =>
     * A -> B
     * C /
     */
    for (auto* inline_node : inline_nodes) {
      // remove this node, merge its inputs to the sink nodes.
      auto inline_inlinks  = inline_node->inlinks();
      auto inline_outlinks = inline_node->outlinks();

      // unlink the inline node from its inputs and outputs
      for (auto& link : inline_inlinks) {
        link->source()->UnLinkTo(link->sink());
      }
      for (auto& link : inline_outlinks) {
        link->source()->UnLinkTo(link->sink());
      }

      // link inline node's input nodes to its output nodes.
      for (auto out_edge : inline_outlinks) {
        auto* out = out_edge->sink();
        for (auto in_edge : inline_inlinks) {
          auto* source = in_edge->source();
          source->LinkTo(out);
        }
      }

      graph->DropNode(inline_node);
    }
  } while (!inline_nodes.empty());

  return graph;
}

void CompuGraphAddCtrlDepLinks(common::Graph* graph, StageMap stages) {
  for (auto& x : graph->nodes()) {
    auto* node = x->safe_as<CompuGraphNode>();
    CHECK(node);
    for (auto& dep : stages[node->tensor]->ctrl_depends()) {
      auto* dep_node = graph->RetriveNode(dep->name);
      if (dep_node) {
        VLOG(3) << "Add control link: " << dep << " -> " << node->id();
        dep_node->LinkTo(node);
      }
    }
  }
}

std::unique_ptr<common::Graph> CreateCompGraph(const std::vector<ir::Tensor>& tensors,
                                               StageMap stages,
                                               bool hide_inline) {
  if (hide_inline) {
    auto graph = CreateCompGraphWithInlineTensorHidden(tensors, stages);
    CompuGraphAddCtrlDepLinks(graph.get(), stages);
    return graph;
  } else {
    auto graph = std::make_unique<common::Graph>();
    std::set<ir::Tensor> visited;
    for (auto& t : tensors) {
      CreateCompGraphWithInlineTensors(graph.get(), t, stages, &visited);
    }
    CompuGraphAddCtrlDepLinks(graph.get(), stages);
    return graph;
  }
}

void LowerImpl::CheckArgsUnique() {
  std::unordered_set<std::string> arg_names;
  for (auto& tensor : tensor_args_) {
    CHECK(!stages_[tensor]->inlined()) << "Inline tensor cannot be argument of function";
    CHECK(!arg_names.count(tensor->name))
        << "The argument of the function, tensor [" << tensor->name << "] duplicates in function " << fn_name_;
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
    VLOG(1) << "tensor argument " << tensor->name << " buffer " << tensor->buffer->name;

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
  std::vector<poly::Stage*> stages;
  std::map<std::string, ir::Tensor> all_tensor_map;
  for (auto& t : CollectAllTensors()) {
    all_tensor_map[t->name] = t;
    if (!stages_[t]->inlined()) stages.push_back(stages_[t]);
  }

  auto deps     = CollectExtraDependencies();
  auto schedule = poly::CreateSchedule(
      stages, poly::ScheduleKind::Poly, std::vector<std::pair<std::string, std::string>>(deps.begin(), deps.end()));

  auto func_body = GenerateFunctionBody(schedule.get());

  std::set<std::string> temp_tensor_names;
  for (auto& t : temp_tensor_args_) temp_tensor_names.insert(t->name);

  auto tensor_map = optim::InitialAssignBuffer(&func_body, stages_, all_tensor_map, comp_graph(), temp_tensor_names);
  // copy the tensor(with buffer assigned) back to func's args.
  {
    for (auto& arg : tensor_args_) {
      if (arg->is_placeholder_node()) continue;
      if (arg->buffer.defined()) continue;
      if (arg->body().As<ir::Call>() && arg->body().type().is_void()) continue;  // extern call
      if (tensor_map.find(arg->name) == tensor_map.end()) {
        LOG(INFO) << "Didn't find arg tensor " << arg->name << "in tensor_map.\n"
                  << "The function is " << fn_name_ << "\nAnd all the arg tensors are:\n";
        for (auto& i : tensor_args_) {
          LOG(INFO) << i->name;
        }
        LOG(FATAL) << "Fatal Error!";
      }
      Reference(&arg)->buffer = tensor_map.at(arg->name)->buffer;
    }
  }

  auto func_args         = GenerateFunctionArgumentList(func_body);
  auto func_temp_tensors = CollectTemporaryTensors();
  std::vector<ir::Buffer> temp_buffers;
  std::unordered_set<std::string> buffer_name_set;
  // TODO(Superjomn) write buffer latter.
  for (auto& t : func_temp_tensors) {
    if (!tensor_map.count(t->name)) continue;
    auto& tt = tensor_map.at(t->name);
    if (tt->buffer.defined() && !buffer_name_set.count(tt->buffer->name)) {
      temp_buffers.push_back(tt->buffer);
      buffer_name_set.insert(tt->buffer->name);
    }
  }

  auto func = ir::_LoweredFunc_::Make(fn_name_, func_args, func_body, temp_buffers);

  // some necessary modification.
  optim::ComputeInlineExpand(&func->body, stages_);
  Target target = cuda_axis_info_.valid() ? common::DefaultNVGPUTarget() : common::DefaultHostTarget();
  auto res      = optim::Optimize(func, target, FLAGS_cinn_runtime_display_debug_info);

  UpdateComputeAtBufferShape(&res, stages_);

  if (cuda_axis_info_.valid()) {
    auto* func           = res.as_lowered_func();
    func->cuda_axis_info = cuda_axis_info_;
  }

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
    for (auto& dep : stages_[cnode->tensor]->ctrl_depends()) {
      deps.emplace(dep->name, cnode->tensor->name);
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

  std::map<std::string, ir::Tensor> global_tensor_map;
  for (auto& group : schedule->groups) {
    CHECK_GT(group.nodes.size(), 0) << "group is empty";
    for (auto& node : group.nodes) {
      if (!tensor_map.count(node->id())) continue;
      auto& tensor = tensor_map[node->id()];
      if (!tensor->has_expression()) continue;
      tuple_to_expr[tensor->name] = tensor->tensor_store_expanded_body();
    }

    Expr group_expr = LowerGroup(group, tuple_to_expr, &global_tensor_map, stages_, &cuda_axis_info_);
    if (group_expr.defined()) {
      VLOG(3) << "group expr:\n" << group_expr;
      exprs.push_back(group_expr);
    }
  }

  Expr body = ir::Block::Make(exprs);
  return body;
}

LowerImpl::LowerImpl(const std::string& fn_name,
                     StageMap stages,
                     const std::vector<Tensor>& tensor_args,
                     const std::vector<Var>& scalar_args,
                     const std::vector<Tensor>& temp_tensor_args)
    : fn_name_(fn_name),
      stages_(stages),
      tensor_args_(tensor_args),
      scalar_args_(scalar_args),
      temp_tensor_args_(temp_tensor_args) {
  {  // Initialize the graph
    std::vector<ir::Tensor> tensors(tensor_args.begin(), tensor_args.end());
    tensors.insert(std::end(tensors), temp_tensor_args.begin(), temp_tensor_args.end());

    compu_graph_ = CreateCompGraph(tensors, stages, false /*inline_hide*/);

    VLOG(1) << "compu_graph:\n" << compu_graph_->Visualize();
  }

  std::vector<poly::Stage*> all_stages;
  for (auto& item : stages_) {
    if (!item.second->inlined()) all_stages.push_back(item.second.get());
  }

  std::map<std::string, poly::Stage*> named_stages, read_caches, write_caches, read_caches_rev, write_caches_rev;
  for (auto* stage : all_stages) {
    named_stages[stage->id()] = stage;
  }
  for (auto* stage : all_stages) {
    if (stage->meta.read_cache_relation) {
      read_caches[stage->id()] = named_stages[stage->meta.read_cache_relation->cache_name];
      read_caches_rev[stage->meta.read_cache_relation->cache_name] = stage;
    }
    if (stage->meta.write_cache_relation) {
      write_caches[stage->id()] = named_stages[stage->meta.write_cache_relation->cache_name];
      write_caches_rev[stage->meta.write_cache_relation->cache_name] = stage;
    }
  }

  for (auto* stage : all_stages) {
    if (stage->tensor()->buffer.get() && (read_caches_rev.count(stage->id()) || write_caches_rev.count(stage->id())) &&
        stage->tensor()->buffer->memory_type == ir::MemoryType::GPUShared) {
      auto sync_threads = Compute(
          {},
          [](const std::vector<Expr>& axis) { return runtime::IntrinsicCall(Void(), "__syncthreads", {}); },
          Context::Global().NewName("syncthreads"));

      stages->Insert(sync_threads, ir::CreateStage(sync_threads).get());
      CHECK_EQ(sync_threads->type(), Void());
      stages[sync_threads]->CtrlDepend(ir::Tensor(stage->tensor()));

      for (auto& compute_at : stage->compute_ats()) {
        stages[sync_threads]->ComputeAt(compute_at.stage.get(), compute_at.level);
      }

      temp_tensor_args_.push_back(sync_threads);

      ir::Tensor this_tensor(read_caches_rev.count(stage->id()) ? read_caches_rev.at(stage->id())->tensor()
                                                                : write_caches_rev.at(stage->id())->tensor());

      for (auto* other : all_stages) {
        if (other->id() != stage->id() && other->tensor()->Uses(this_tensor)) {
          other->CtrlDepend(sync_threads);
        }
      }
    }
  }

  {  // update schedule.
    std::vector<ir::Tensor> tensors(tensor_args.begin(), tensor_args.end());
    tensors.insert(std::end(tensors), temp_tensor_args_.begin(), temp_tensor_args_.end());
    compu_graph_ = CreateCompGraph(tensors, stages, true /*inline_hide*/);

    VLOG(1) << "Computation Graph:\n" << compu_graph_->Visualize();
  }
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn

#include "cinn/lang/lower_impl.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/tensor.h"
#include "cinn/optim/ir_replace.h"
#include "cinn/poly/compute_at_transform.h"

namespace cinn {
namespace lang {
namespace detail {

using ir::ComputeAtInfo;

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
  // now we get a workable expression, but the statement are something like `B(((16 * po0) + po1), po2)`, we need to
  // transform this to some realworld statement in CINN.

  VLOG(1) << "ast to expr: \n" << e << std::endl;

  // replace isl call to the corresponding CINN statement, we need to replace the axis at the same time.
  for (auto& statement : tuple_to_expr) {
    LOG(INFO) << "working on statement: " << statement.first;
    if (!gen.ContainsStatement(statement.first)) continue;
    // the axis_ast_map contains the axis from the original (like `i`) to the transformed (like `i+3`).
    auto axis_expr_map = gen.axis2expr(statement.first);
    for (auto& item : axis_expr_map) {
      LOG(INFO) << "axis: " << item.first << " " << item.second;
    }

    // the original CINN statements.
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    VLOG(3) << "replacing " << statement.first << " to " << statement_candi_expr;
    optim::ReplaceCallWithExpr(&e, statement.first, statement_candi_expr, axis_expr_map);
  }
  CheckNoIslCallRemains(&e);

  // deal with the compute_at relations
  ProcessComputeAtInfo(&e);

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

  common::UnifyAllTensorsInExpr(&res);
  common::UnifyAllBuffersInExpr(&res);

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

void ModifyProducerStatementIndices(const std::vector<Var>& consumer_forloop_iters,
                                    Expr* consumer_forloop_root,
                                    ir::ComputeAtInfo info);

/**
 * Lets define the consumer tensor as C and the producer tensor as P for short.
 * First, find the forloop generating C, keep the forloop levels in a stack.
 * We need to modify the following
 * 1. P's Store indice(change the parameters to zero)
 * 2. P's Store value, change the parameters in Load to consumer's precending axis
 * 3. replace the precending axis of the P's Load to zero in C
 */
struct CorrectComputeAtRelatedIndiceMutator : public ir::IRMutator<> {
  std::string tensor_name;

  CorrectComputeAtRelatedIndiceMutator(const std::string& tensor_name) : tensor_name(tensor_name) {}

  void operator()(Expr* e) { return ir::IRMutator<>::Visit(e, e); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  void Visit(const ir::For* op, Expr* expr) override {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  void ReplaceParamWithConsumerAxis(const ComputeAtInfo& info,
                                    const std::vector<Var>& axis,
                                    Expr* consumer_forloop_root) {
    CHECK_LE(info.level + 1, axis.size());
    // replace the params to consumer's precending level+1 axis.
    for (int i = 0; i < info.level + 1; i++) {
      Var var(poly::GenConsumerParamName(info.consumer_tensor_name.c_str(), i));
      LOG(INFO) << "replacing " << var << " to " << axis[i];
      optim::IrReplace(consumer_forloop_root, Expr(var), axis[i]);
    }
  }

  //! Get a stack of forloops to a Store node target to \p tensor_name
  std::vector<Expr*> GetForloopStackToStore(Expr* expr, const std::string& tensor_name) {
    struct Mutator : public ir::IRMutator<> {
      std::vector<Expr*> forloop_stack;
      bool found{false};

      std::string tensor_name;

      Mutator(const std::string& tensor_name) : tensor_name(tensor_name) {}

      std::vector<Expr*> operator()(Expr* expr) {
        ir::IRMutator<>::Visit(expr, expr);
        return forloop_stack;
      }

      void Visit(const ir::For* op, Expr* expr) {
        auto* node = expr->As<ir::For>();
        forloop_stack.push_back(expr);
        ir::IRMutator<>::Visit(&node->body, &node->body);
        if (!found) forloop_stack.pop_back();
      }

      void Visit(const ir::PolyFor* op, Expr* expr) {
        auto* node = expr->As<ir::PolyFor>();
        forloop_stack.push_back(expr);
        ir::IRMutator<>::Visit(&node->body, &node->body);
        if (!found) forloop_stack.pop_back();
      }

      void Visit(const ir::Store* op, Expr* expr) { found = op->tensor.as_tensor()->name == tensor_name; }
    };

    return Mutator(tensor_name)(expr);
  }

  /**
   * Normalize the producer's domain, make it start from zero. This is essential for shink the buffer and inference the
   * buffer size.
   *
   * e.g.
   * for (i=p0; i<3+p0; i++) {
   *   p[i]
   * }
   * will be transformed to
   * for (i=0; i<3; i++) {
   *   p[i+p0]
   * }
   *
   * @param producer_forloop_root The root of the producer's own axis, not the axis of consumer.
   *
   * About the \p producer_forloop_root, after compute_at schedule,
   * // consumer iter ci
   * for (ci) {
   *   // producer iter pi
   *   for (pi) {
   *   }
   * }
   * The pi should be the \p producer_forloop_root
   */
  void NormalizeProducerDomain(Expr* producer_forloop_root,
                               const std::string& producer_tuple,
                               const std::vector<Var>& consumer_axis) {
    LOG(INFO) << "Normalize producer domain: " << producer_tuple;
    LOG(INFO) << "producer_forloop_root:\n" << *producer_forloop_root;
    LOG(INFO) << "consumer_axis:";
    for (auto& var : consumer_axis) {
      LOG(INFO) << "iter: " << var;
    }

    struct Mutator : public ir::IRMutator<> {
      std::map<Var, Expr> offsets;
      std::vector<Var> consumer_axis;
      std::string producer_tuple;

      Mutator(const std::string& producer_tuple, const std::vector<Var>& consumer_axis)
          : producer_tuple(producer_tuple), consumer_axis(consumer_axis) {}

      void operator()(Expr* forloop) { ir::IRMutator<>::Visit(forloop, forloop); }

      //! Add offsets to store, e.g. offset is i->3, the original store expr is a[i,j] = b[i*2,j], the result expression
      //! will be a[i+3,j] = b[(i+3)*2,j]
      void AddOffsetsToStoreExpr(Expr* expr) {
        CHECK(expr->As<ir::Store>());
        for (auto& offset : offsets) {
          optim::IrReplace(expr, offset.first, Expr(offset.first) + offset.second);
        }
      }

      //! Set the producer axis to zero in Store node, e.g. a store node, a[c0,c1] = ... will be a[0,0]
      void SetProducerAxisToZeroInStore(Expr* expr) {
        auto* node = expr->As<ir::Store>();
        CHECK(node);

        for (auto& indice : node->indices) {
          for (auto& consumer_axis : consumer_axis) {
            optim::IrReplace(&indice, consumer_axis, common::make_const(0));
          }
        }
      }

      //! NOTE the axis here should be producer's axis, `i` in the root function comment.
      void AddOffsetToAxisInStoreValue(Expr* expr) {
        auto* node = expr->As<ir::Store>();

        auto loads_but_producer = ir::CollectIRNodes(node->value, [&](const Expr* x) {
          return x->As<ir::Load>() && x->As<ir::Load>()->tensor.as_tensor()->name != node->tensor.as_tensor()->name;
        });

        for (auto& item : loads_but_producer) {
          auto* load = item.As<ir::Load>();
          for (auto& indice : load->indices) {
            for (auto& offset : offsets) {
              optim::IrReplace(&Reference(&indice), offset.first, Expr(offset.first) + offset.second);
            }
          }
        }
      }

      void Visit(const ir::Store* op, Expr* expr) override {
        auto* node = expr->As<ir::Store>();

        if (op->tensor.as_tensor()->name == producer_tuple) {
          AddOffsetsToStoreExpr(expr);

          // replace the producer axis in store indice to zero.
          SetProducerAxisToZeroInStore(expr);

          // replace the consumer axis in value(not producer) to offset.
          AddOffsetToAxisInStoreValue(expr);
        } else {
          ir::IRMutator<>::Visit(op, expr);
        }
      }

      void Visit(const ir::For* op, Expr* expr) override {
        auto* node = expr->As<ir::For>();
        if (!common::is_zero(op->min)) {
          auto offset             = op->min;
          node->min               = common::make_const(0);
          node->extent            = node->extent - offset;
          offsets[node->loop_var] = offset;
        }
        ir::IRMutator<>::Visit(&node->body, &node->body);
      }

      void Visit(const ir::PolyFor* op, Expr* expr) override {
        auto* node = expr->As<ir::PolyFor>();
        if (!common::is_zero(op->init)) {
          auto offset = op->init;
          node->init  = common::make_const(0);
          UpdatePolyForConditionWithOffset(&node->condition, node->iterator, offset);
        }
        ir::IRMutator<>::Visit(&node->body, &node->body);
      }

      void UpdatePolyForConditionWithOffset(Expr* cond, Var iter, Expr offset) {
        optim::IrReplace(cond, iter, Expr(iter) + offset);
      }
    };

    Mutator(producer_tuple, consumer_axis)(producer_forloop_root);
  }

  //! Reset the indice of the producer Load in Consumer.
  // Here we just set the consumer axis to zero.
  void ResetConsumerLoadIndice(const std::vector<Var>& consumer_axis,
                               Expr* consumer_store_expr,
                               const std::string& producer_tensor_name) {
    struct Mutator : public ir::IRMutator<> {
      const std::string& producer_tensor_name;
      const std::vector<Var>& consumer_axis;

      Mutator(const std::string& producer_tensor_name, const std::vector<Var>& consumer_axis)
          : producer_tensor_name(producer_tensor_name), consumer_axis(consumer_axis) {}

      void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

      void Visit(const ir::Load* op, Expr* expr) override {
        auto* node = expr->As<ir::Load>();
        if (op->tensor.as_tensor()->name == producer_tensor_name) {
          for (auto axis : consumer_axis) {
            for (auto& indice : node->indices) {
              optim::IrReplace(&indice, axis, common::make_const(0));
            }
          }
        }
        // Load not recursive, no need to visit it's items.
      }
    };

    Mutator(producer_tensor_name, consumer_axis)(consumer_store_expr);
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();

    if (op->tensor.as_tensor()->name != tensor_name) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }

    // get the target consumer
    auto& compute_at_infos = op->tensor.as_tensor()->compute_at_infos;
    CHECK(!compute_at_infos.empty());

    std::vector<Var> levels;
    for (Expr* forloop : forloop_stack) {
      auto* for_n      = forloop->As<ir::For>();
      auto* poly_for_n = forloop->As<ir::PolyFor>();
      if (for_n)
        levels.push_back(for_n->loop_var);
      else if (poly_for_n)
        levels.push_back(poly_for_n->iterator);
      else
        NOT_IMPLEMENTED
    }

    for (auto& compute_at_info : compute_at_infos) {
      LOG(INFO) << "compute_at: " << compute_at_info.producer_tensor_name;
      ReplaceParamWithConsumerAxis(compute_at_info, levels, forloop_stack.front());
    }

    LOG(INFO) << "forloop_stack: =====================";
    for (auto& e : forloop_stack) {
      LOG(INFO) << "--------------------------------";
      LOG(INFO) << *e;
    }

    for (auto& compute_at_info : compute_at_infos) {
      int level = compute_at_info.level;
      std::vector<Var> consumer_aixs(levels.begin(), levels.begin() + level + 1);
      Expr* producer_forloop_root;
      if (forloop_stack[level]->As<ir::For>()) {
        producer_forloop_root = &forloop_stack[level]->As<ir::For>()->body;
      } else {
        producer_forloop_root = &forloop_stack[level]->As<ir::PolyFor>()->body;
      }

      auto forloop_stack_to_store = GetForloopStackToStore(producer_forloop_root, compute_at_info.producer_tensor_name);
      CHECK(!forloop_stack_to_store.empty());
      producer_forloop_root = forloop_stack_to_store.front();

      NormalizeProducerDomain(producer_forloop_root, compute_at_info.producer_tensor_name, consumer_aixs);
      ResetConsumerLoadIndice(consumer_aixs, forloop_stack[level + 1], compute_at_info.producer_tensor_name);
    }
  }

  /**
   * Modify the producer statement indice.
   * @param consumer_forloop_iters The iters of the forloop tree.
   * @param consumer_forloop_root The root of the forloop tree.
   */
  void ModifyProducerStatementIndices(const std::vector<Var>& consumer_forloop_iters,
                                      Expr* consumer_forloop_root,
                                      ir::ComputeAtInfo info) {
    CHECK(consumer_forloop_root->As<ir::For>());

    struct Mutator : public ir::IRMutator<> {
      ir::ComputeAtInfo info;
      Mutator(ir::ComputeAtInfo info) : info(info) {}

      void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

      void Visit(const ir::Store* op, Expr* expr) {
        auto* node = expr->As<ir::Store>();
        if (op->tensor.as_tensor()->name == info.producer_tensor_name) {
        } else {
          ir::IRMutator<>::Visit(op, expr);
        }
      }
    };
  }

  std::vector<Expr*> forloop_stack;
};

void ProcessComputeAtInfo(Expr* expr) {
  // 1. collect all the consumer tensors thouse have compute_at_infos.
  // 2. for each producer tensor, reset the producer tensor loads indice.

  // first, visit the consumer tensor with compute_at info.
  // second, in the forloop stack, find the producer tensor
  //    - set the presending axis to zero in producer's Store node and Load node
  //    - replace the ISL parameter to the precending axis
  // in consumer, reset presending axis in producer's Load to zero.

  auto tensor_with_compute_at_infos = ir::CollectIRNodes(
      *expr, [&](const Expr* x) { return x->as_tensor() && !x->as_tensor()->compute_at_infos.empty(); });

  for (auto& tensor : tensor_with_compute_at_infos) {
    LOG(INFO) << "consumer: " << tensor;
    CorrectComputeAtRelatedIndiceMutator(tensor.as_tensor()->name)(expr);
  }
}

void ResizeComputeAtBuffer(Expr* expr) {
  auto tensor_with_compute_at_infos = ir::CollectIRNodes(*expr, [&](const Expr* x) {
    return x->as_tensor() && !x->as_tensor()->inlined() && !x->as_tensor()->compute_at_infos.empty();
  });

  auto tensor_map = ir::CollectTensorMap(*expr, [&](const Expr* x) { return !x->as_tensor()->inlined(); });

  std::unordered_map<std::string, ir::ComputeAtInfo*> buffer_to_compute_at_info;
  for (auto& item : tensor_map) {
    auto& compute_at_infos = item.second.as_tensor()->compute_at_infos;
    if (compute_at_infos.empty()) continue;
    for (auto& compute_at : compute_at_infos) {
      auto& producer_tensor = tensor_map.at(compute_at.producer_tensor_name);
      buffer_to_compute_at_info[producer_tensor.as_tensor()->buffer->name] = &compute_at_infos.front();
    }
  }

  auto tensors = ir::CollectIRNodes(*expr, [&](const Expr* x) { return x->as_tensor() && !x->as_tensor()->inlined(); });
  for (auto& t : tensors) {
    if (!buffer_to_compute_at_info.count(t.as_tensor()->buffer->name)) continue;
    auto& buffer       = t.as_tensor()->buffer;
    auto compute_at_it = buffer_to_compute_at_info.find(buffer->name);
    if (compute_at_it != buffer_to_compute_at_info.end()) {
      // process_tensor(&Reference(t.as_tensor()), *compute_at_it->second);
      // process_buffer(Reference(t.as_tensor()).buffer, *compute_at_it->second);
      LOG(INFO) << "*resizing buffer " << t;
      LOG(INFO) << "*resizing tensor " << t.as_tensor()->buffer;
    }
  }

  auto loads = ir::CollectIRNodes(*expr, [&](const Expr* x) { return x->As<ir::Load>(); });
  for (auto& item : loads) {
    LOG(INFO) << "load: " << item << " index: " << item.As<ir::Load>()->index()
              << " buffer: " << item.As<ir::Load>()->tensor.as_tensor()->buffer;
  }
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn

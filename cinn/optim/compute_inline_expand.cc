#include "cinn/optim/compute_inline_expand.h"

#include "cinn/common/graph_utils.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

/*
 * Replace a tensor(marked as compute_inline) to the expanded expression.
 */
struct TensorInlineExpandMutator : public ir::IRMutator<> {
  const std::string &tensor_name;

  TensorInlineExpandMutator(const std::string &tensor_name) : tensor_name(tensor_name) {}

  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto *node   = expr->As<ir::Load>();
    auto *tensor = node->tensor.as_tensor();
    if (tensor && tensor->name == tensor_name) {
      *expr = tensor->inline_expanded(op->indices);
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
    }
  }
};

struct SSANode : public common::GraphNode {
  std::string id_;

  SSANode(const std::string &id) : id_(id) {}

  std::string id() const override { return id_; }

  const char *type_info() const override { return __type_info__; }

  static constexpr char *__type_info__ = "optim::SSANode";
};

// TODO(Superjomn) the graph here is not a SSA now, it is flattern for the ir::CollectIRNodes method collects all the
// tensors recursively, so it can not reserve the level information, fix it.
struct SSABuilder : public ir::IRMutator<> {
  common::Graph graph;

  SSABuilder &operator()(Expr *expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return *this;
  }

  void Visit(const ir::Store *op, Expr *expr) override {
    LOG(INFO) << "Expr: " << *expr;
    auto *node = expr->As<ir::Store>();

    auto *cur_graph_node = graph.RetriveNode(node->tensor.as_tensor()->name);
    if (!cur_graph_node) {
      cur_graph_node = graph.RegisterNode(node->tensor.as_tensor()->name, new SSANode(node->tensor.as_tensor()->name));
    }

    auto deps_tensor_names = node->tensor.as_tensor()->GetDependTensorNames();
    for (auto &t : deps_tensor_names) {
      auto *n = graph.RetriveNode(t);
      if (!n) {
        n = graph.RegisterNode(t, new SSANode(t));
      }

      n->LinkTo(cur_graph_node);
    }
  }
};

}  // namespace

void ComputeInlineExpand(Expr *expr, poly::StageMap stages) {
  /*
   * 1. Create a SSA graph
   * 2. follow the topological order, expand inline each tensor marked as compute_inline.
   */

  SSABuilder ssa_builder;
  auto &ssa_graph = ssa_builder(expr).graph;
  VLOG(4) << "inline SSA graph:\n" << ssa_graph.Visualize();
  LOG(INFO) << "inline SSA graph:\n" << ssa_graph.Visualize();

  auto [node_order, edge_order] = ssa_graph.topological_order();  // NOLINT

  auto tensors = ir::CollectIRNodes(*expr, [](const Expr *x) { return x->as_tensor(); });
  std::map<std::string, ir::Tensor> tensor_map;
  for (auto &x : tensors) {
    auto t              = x.as_tensor_ref();
    tensor_map[t->name] = t;
  }

  for (auto *n : node_order) {
    auto *node = n->safe_as<SSANode>();
    auto t     = tensor_map.at(node->id());
    if (stages[t]->inlined()) {
      VLOG(2) << "inlining " << t->name;
      TensorInlineExpandMutator(t->name)(expr);
    }
  }
}

}  // namespace optim
}  // namespace cinn
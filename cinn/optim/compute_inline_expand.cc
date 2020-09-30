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
  // the inline tensors contained in the expression.
  auto inline_tensors =
      ir::CollectIRNodes(*expr, [&](const Expr *x) { return x->as_tensor() && stages[x->as_tensor()]->inlined(); });

  // keep inline expand if any inline tensor exists
  // NOTE This is a naive method to greedily expand the inline tensors until none exists, a better way is to create a
  // SSA graph and expand the inline tensors in the reversed dependency order.
  // TODO(Superjomn) Use the SSA graph to improve this.
  while (!inline_tensors.empty()) {
    for (const auto &t : inline_tensors) {
      auto *tensor = t.as_tensor();
      TensorInlineExpandMutator(tensor->name)(expr);
    }

    inline_tensors = ir::CollectLoadTensors(
        *expr, [&](const Expr *x) { return x->as_tensor() && stages[x->as_tensor()]->inlined(); });
  }
}

}  // namespace optim
}  // namespace cinn
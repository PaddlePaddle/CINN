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
  std::map<std::string, ir::Tensor> *all_tensor_map_;
  bool inline_code{false};
  bool temp_buffer{false};
  bool memory_local{false};

  TensorInlineExpandMutator(const std::string &tensor_name, std::map<std::string, ir::Tensor> *all_tensor_map)
      : tensor_name(tensor_name), all_tensor_map_(all_tensor_map) {}

  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *expr, Expr *op) override {
    if (inline_code && temp_buffer) {
      if (utils::Startswith(expr->name, "blockIdx") || (utils::Startswith(expr->name, "threadIdx") && memory_local)) {
        *op = ir::Expr(0);
      }
    }
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto *node   = expr->As<ir::Load>();
    auto *tensor = node->tensor.as_tensor();
    if (tensor && tensor->name == tensor_name) {
      *expr       = tensor->inline_expanded(op->indices);
      inline_code = true;
      ir::IRMutator<>::Visit(expr, expr);
      inline_code = false;
    } else if (inline_code && tensor->buffer.defined() &&
               (utils::Endswith(tensor->buffer->name, "_read_cache") ||
                utils::Endswith(tensor->buffer->name, "_cache_write_out") ||
                utils::Endswith(tensor->buffer->name, "_temp_buffer"))) {
      bool keep_buffer       = temp_buffer;
      temp_buffer            = true;
      bool keep_memory_local = memory_local;
      if ((*all_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal) {
        memory_local = true;
      }
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto &idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
      temp_buffer  = keep_buffer;
      memory_local = keep_memory_local;
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto &idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
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
    auto *node = expr->As<ir::Store>();

    auto *cur_graph_node = graph.RetrieveNode(node->tensor.as_tensor()->name);
    if (!cur_graph_node) {
      cur_graph_node = graph.RegisterNode(node->tensor.as_tensor()->name, new SSANode(node->tensor.as_tensor()->name));
    }

    auto deps_tensor_names = node->tensor.as_tensor()->GetDependTensorNames();
    for (auto &t : deps_tensor_names) {
      auto *n = graph.RetrieveNode(t);
      if (!n) {
        n = graph.RegisterNode(t, new SSANode(t));
      }

      n->LinkTo(cur_graph_node);
    }
  }
};

}  // namespace

void ComputeInlineExpand(Expr *expr, poly::StageMap stages, std::map<std::string, ir::Tensor> *all_tensor_map) {
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
      TensorInlineExpandMutator(tensor->name, all_tensor_map)(expr);
    }

    inline_tensors = ir::CollectLoadTensors(
        *expr, [&](const Expr *x) { return x->as_tensor() && stages[x->as_tensor()]->inlined(); });
  }
}

}  // namespace optim
}  // namespace cinn
#include "cinn/optim/buffer_assign.h"

#include "cinn/common/union_find.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower_impl.h"
#include "cinn/optim/ir_replace.h"

namespace cinn {
namespace optim {

namespace {

struct BufferUFNode : public common::UnionFindNode {
  BufferUFNode(const std::string& x) : tensor_name(x) {}

  const char* type_info() const override { return __type_info__; }

  std::string tensor_name;
  static const char* __type_info__;
};

const char* BufferUFNode::__type_info__ = "BufferUFNode";

struct IRReplaceTensorMutator : ir::IRMutator<> {
  const std::map<std::string, ir::Tensor>& tensor_map;
  IRReplaceTensorMutator(const std::map<std::string, ir::Tensor>& tensor_map) : tensor_map(tensor_map) {}
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto it = tensor_map.find(op->name);
    if (it != tensor_map.end()) {
      *expr = Expr(it->second);
    }
  }
};

}  // namespace

std::map<std::string, ir::Tensor> InitialAssignBuffer(Expr* expr,
                                                      poly::StageMap stages,
                                                      const std::map<std::string, ir::Tensor>& all_tensor_map,
                                                      const common::Graph* comp_graph,
                                                      const std::set<std::string>& temp_tensor_names) {
  // The tensor map helps to reserve only one tensor instance for a tensor(called the same name).
  std::map<std::string, ir::Tensor> buffer_updated_tensor;

  for (auto& item : all_tensor_map) {
    if (stages[item.second]->inlined()) continue;
    buffer_updated_tensor[item.second->name] = item.second;
  }

  // union-find to cluster the tensors with the same buffer.
  common::UnionFind union_find;

  // unify all the tensor occurance with a global one, e.g. there are multiple tensor B exists in the expression,
  // replace them with a shared one.
  ir::CollectIRNodes(*expr, [&](const Expr* x) -> bool {
    auto* t = x->as_tensor();
    if (t && !stages[t]->inlined()) {
      Reference(x) = Expr(all_tensor_map.at(t->name));
    }
    return false;
  });

  std::map<std::string, BufferUFNode*> uf_map;
  for (auto& item : all_tensor_map) {
    auto* n                   = union_find.AddNode(new BufferUFNode(item.second->name));
    uf_map[item.second->name] = n->safe_as<BufferUFNode>();
  }

  for (auto& item : buffer_updated_tensor) {
    auto* cur_n = uf_map[item.first];
    for (auto& other : stages[item.second]->meta.tensors_to_share_buffer_with) {
      // we might intialize the buffer in args.
      auto* other_n = uf_map[other];
      if (!other_n) continue;

      VLOG(3) << "share buffer between " << item.first << " " << other_n->tensor_name;
      cur_n->Union(other_n);
    }
  }

  // determine which tensor to have the initial buffer, and will share accross the cluser, we take a topological order
  // of the computational graph, and find out which tensor comes first in a cluster.

  auto [topo_order, topo_edges] = comp_graph->topological_order();
  for (common::GraphNode* n : topo_order) {
    auto nn = n->safe_as<lang::detail::CompuGraphNode>();
    CHECK(nn);
    {
      auto it = uf_map.find(nn->tensor->name);
      CHECK(it != uf_map.end());
      auto& cluster_info = std::get<0>(it->second->GetRoot())->cluster_info;
      if (cluster_info.empty()) {  // buffer owner(a tensor) of this cluster not set yet.
        cluster_info = nn->tensor->name;
      }
    }
  }

  // Get a center of the cluster, it will consider the following rules
  // 1. Prefer a tensor arg than a temp tensor.
  auto cluster_get_center_tensor = [&](const std::vector<common::UnionFindNode*>& cluster) {
    ir::Tensor some_tensor;
    // try to find a node that is a tensor_arg, allocate buffer for it, and make others share buffer with it.
    for (auto* n : cluster) {
      auto* node   = n->safe_as<BufferUFNode>();
      bool is_temp = temp_tensor_names.count(node->tensor_name);
      if (!is_temp) return all_tensor_map.at(node->tensor_name);
      some_tensor = all_tensor_map.at(node->tensor_name);
    }
    return some_tensor;
  };

  for (auto& cluster : union_find.GetClusters()) {
    auto root_tensor = cluster_get_center_tensor(cluster);
    if (!root_tensor->buffer.defined() && !root_tensor->type().is_void()) root_tensor->WithBuffer();

    for (auto* n : cluster) {
      auto& tensor = all_tensor_map.at(n->safe_as<BufferUFNode>()->tensor_name);
      if (tensor != root_tensor) {
        Reference(&tensor)->Bind(root_tensor->buffer);
        VLOG(3) << "tensor " << tensor->name << " bind buffer [" << tensor->buffer->name << "]";
      }
    }
  }

  return buffer_updated_tensor;
}

}  // namespace optim
}  // namespace cinn
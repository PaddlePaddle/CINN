#include "cinn/optim/buffer_assign.h"

#include "cinn/common/union_find.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
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
  void operator()(Expr* expr) {
    LOG(INFO) << "original expr: " << *expr;
    ir::IRMutator<>::Visit(expr, expr);
  }

  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto it = tensor_map.find(op->name);
    if (it != tensor_map.end()) {
      LOG(INFO) << "unify tensor " << *expr;
      *expr = Expr(it->second);
      LOG(INFO) << "unified to " << expr->as_tensor();
    }
  }
};

}  // namespace

std::map<std::string, ir::Tensor> InitialAssignBuffer(Expr* expr,
                                                      poly::StageMap stages,
                                                      const std::vector<std::set<std::string>>& buffer_shared) {
  std::map<std::string, ir::Tensor> tensor_map;

  auto tensor_exprs = ir::CollectIRNodes(*expr, [&](const Expr* x) {
    auto* t = x->as_tensor();
    return t && (!stages[t]->meta.compute_inline) && !t->buffer.defined();
  });

  if (tensor_exprs.empty()) return tensor_map;

  // union-find to cluster the tensors with the same buffer.
  common::UnionFind union_find;

  // The tensor map helps to reserve only one tensor instance for a tensor(called the same name).
  for (auto& e : tensor_exprs) {
    tensor_map[e.as_tensor()->name] = e.as_tensor_ref();
  }
  // unify all the tensor occurance with a global one, e.g. there are multiple tensor B exists in the expression,
  // replace them with a shared one.
  ir::CollectIRNodes(*expr, [&](const Expr* x) -> bool {
    auto* t = x->as_tensor();
    if (t && tensor_map.count(t->name)) {
      Reference(x) = Expr(tensor_map.at(t->name));
    }
    return false;
  });

  auto existing_tensors = ir::CollectIRNodes(*expr, [&](const Expr* x) {
    auto* t = x->as_tensor();
    return t && !stages[t]->meta.compute_inline && !t->buffer.defined();
  });
  CHECK_EQ(existing_tensors.size(), tensor_map.size())
      << "some of the tensors named same are not unified to one object";

  std::map<std::string, BufferUFNode*> uf_map;
  for (auto& item : tensor_map) {
    auto* n                   = union_find.AddNode(new BufferUFNode(item.second->name));
    uf_map[item.second->name] = n->safe_as<BufferUFNode>();
  }

  for (auto& item : tensor_map) {
    auto* cur_n = uf_map[item.first];
    if (!stages[item.second]->meta.tensors_to_share_buffer_with.empty()) {
      for (auto& other : stages[item.second]->meta.tensors_to_share_buffer_with) {
        // we might intialize the buffer in args.
        auto* other_n = uf_map[other];
        if (!other_n) continue;

        VLOG(3) << "share buffer between " << item.first << " " << other_n->tensor_name;
        cur_n->Union(other_n);
      }
    }
  }

  for (auto& cluster : union_find.GetClusters()) {
    VLOG(5) << "get cluster size " << cluster.size();
    auto& first_tensor = tensor_map.at(cluster[0]->safe_as<BufferUFNode>()->tensor_name);
    first_tensor->WithBuffer();
    VLOG(3) << "first_tensor: " << first_tensor->name << " buffer " << first_tensor->buffer;
    for (int i = 1; i < cluster.size(); i++) {
      auto& tensor = tensor_map.at(cluster[i]->safe_as<BufferUFNode>()->tensor_name);
      tensor->Bind(first_tensor->buffer);
      VLOG(3) << "tensor [" << tensor->name << "] bind buffer [" << first_tensor->buffer << "]";
    }
  }

  return tensor_map;
}

}  // namespace optim
}  // namespace cinn
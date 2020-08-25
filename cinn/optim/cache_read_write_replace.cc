#include "cinn/optim/cache_read_write_replace.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

namespace {

/**
 * Replace the reader of a cache tensor to tensor.
 */
struct CacheReplaceMutator : public ir::IRMutator<> {
  std::string tensor_name;
  ir::Tensor cache;
  std::set<std::string> cache_reader_names;
  bool read_or_write{};

  /**
   * construct
   * @param tensor_name name of the tensor to cache
   * @param cache the cache
   * @param cache_reader_names names of the cache readers
   * @param read_or_write read or write cache
   */
  CacheReplaceMutator(const std::string& tensor_name,
                      ir::Tensor cache,
                      const std::vector<std::string>& cache_reader_names,
                      bool read_or_write)
      : tensor_name(tensor_name),
        cache(cache),
        cache_reader_names(cache_reader_names.begin(), cache_reader_names.end()),
        read_or_write(read_or_write) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node            = expr->As<ir::Store>();
    bool read_cache_match = op->tensor.as_tensor() && cache_reader_names.count(op->tensor.as_tensor()->name);

    if (read_cache_match) {
      {
        to_mutate_ = read_cache_match;

        ir::IRMutator<>::Visit(&node->value, &node->value);

        to_mutate_ = false;
      }

    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      ir::IRMutator<>::Visit(&node->value, &node->value);
    }
  }

  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    if (to_mutate_ && tensor_name == op->name) {
      *expr = cache;
    }
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto* node   = expr->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (to_mutate_ && tensor && tensor->name == cache->name) {
      node->tensor = Expr(cache);
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
    }
  }

  bool to_mutate_{false};
};

}  // namespace

void CacheReadWriteReplace(Expr* expr, poly::StageMap stages, std::map<std::string, ir::Tensor>* global_tensor_map) {
  auto cached_tensors = ir::CollectIRNodes(*expr, [&](const Expr* x) {
    auto* t = x->as_tensor();
    return t && (stages[t]->meta.read_cache_relation || stages[t]->meta.write_cache_relation);
  });

  std::set<ir::Tensor> uniq_cached_tensors;
  for (auto& x : cached_tensors) {
    uniq_cached_tensors.insert(x.as_tensor_ref());
  }

  std::map<std::string, ir::Tensor> tensor_map;
  for (auto& e : stages) {
    auto t                     = ir::Tensor(e.second->tensor());
    tensor_map[e.second->id()] = t;
  }

  // update global_tensor_map
  for (auto& item : tensor_map) {
    if (!global_tensor_map->count(item.first)) {
      (*global_tensor_map)[item.first] = item.second;
    }
  }

  for (auto& t : uniq_cached_tensors) {
    if (stages[t]->meta.read_cache_relation) {
      auto cache = global_tensor_map->at(stages[t]->meta.read_cache_relation->cache_name);
      CacheReplaceMutator(t->name, cache, stages[t]->meta.read_cache_relation->readers, true /*read*/)(expr);
    }
    if (stages[t]->meta.write_cache_relation) {
      auto cache = tensor_map.at(stages[t]->meta.write_cache_relation->cache_name);
      CacheReplaceMutator(t->name, cache, {}, false /*write*/)(expr);
    }
  }
}

}  // namespace optim
}  // namespace cinn

#include "cinn/optim/replace_var_with_expr.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/tensor.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprMutator : public ir::IRMutator<> {
  ReplaceVarWithExprMutator(const Var& var, const Expr& expr) : var_(var), expr_(expr) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name != var_->name) return;
    auto copied = IRCopy(expr_);
    *op         = copied;
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->min, &node->min);
    ir::IRMutator<>::Visit(&node->extent, &node->extent);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->loop_var->name == var_->name && expr_.As<ir::_Var_>()) {
      node->loop_var = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->init, &node->init);
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    ir::IRMutator<>::Visit(&node->inc, &node->inc);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->iterator->name == var_->name && expr_.As<ir::_Var_>()) {
      node->iterator = expr_.As<ir::_Var_>();
    }
  }

 private:
  const Var& var_;
  const Expr& expr_;
};

void ReplaceVarWithExpr(Expr* source, const Var& var, const Expr& expr) {
  ReplaceVarWithExprMutator mutator(var, expr);
  mutator(source);
}

struct ReplaceVarIndexOfCacheMutator : public ir::IRMutator<> {
  ReplaceVarIndexOfCacheMutator(const Var& var,
                                const Expr& expr,
                                std::map<std::string, ir::Tensor>* global_tensor_map,
                                std::unordered_set<std::string>* resized_buffer,
                                bool blockidx,
                                const Expr& extent)
      : var_(var),
        expr_(expr),
        global_tensor_map_(global_tensor_map),
        resized_buffer_(resized_buffer),
        blockidx_(blockidx),
        extent_(extent) {}

  void Execute(Expr* expr) {
    auto* for_     = expr->As<ir::For>();
    auto* poly_for = expr->As<ir::PolyFor>();
    if (for_) {
      ir::IRMutator<>::Visit(&for_->body, &for_->body);
    } else {
      ir::IRMutator<>::Visit(&poly_for->body, &poly_for->body);
    }
  }

  void ResizeTempMemory(const std::string& tensor_name) {
    if (extent_.defined()) {
      std::string buffer_id = (*global_tensor_map_)[tensor_name]->buffer->name + var_->name;
      if (resized_buffer_->count(buffer_id) != 0) return;
      auto buffer_shape = (*global_tensor_map_)[tensor_name]->buffer->shape;
      Expr prod(1);
      for (auto i : buffer_shape) {
        prod = ir::Mul::Make(prod, i);
      }
      prod = ir::Div::Make(prod, extent_);
      std::vector<Expr> new_shape{prod};
      (*global_tensor_map_)[tensor_name]->buffer->shape = new_shape;
      resized_buffer_->insert(buffer_id);
    } else {
      VLOG(3) << "extent not defined";
    }
    return;
  }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (do_replace) {
      if (expr->name != utils::GetStreamCnt(var_->name)) return;
      VLOG(2) << "Do Replace: " << expr->name << " to 0";
      auto copied  = IRCopy(expr_);
      *op          = copied;
      find_replace = true;
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node   = expr->As<ir::Store>();
    auto* tensor = node->tensor.as_tensor();
    VLOG(2) << "Store 's tensor name is : " << tensor->name;
    if (tensor->buffer.defined() &&
        (utils::Endswith(tensor->buffer->name, "_read_cache") ||
         utils::Endswith(tensor->buffer->name, "_cache_write_out")) &&
        ((*global_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal || blockidx_)) {
      bool temp_replace = do_replace;
      do_replace        = true;
      find_replace      = false;
      for (auto& index : node->indices) {
        ir::IRMutator<>::Visit(&index, &index);
      }
      if (find_replace == true) ResizeTempMemory(tensor->name);
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      do_replace = temp_replace;
    } else {
    }
    ir::IRMutator<>::Visit(&node->value, &node->value);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node   = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    VLOG(2) << "Load's tensor name is : " << tensor->name;
    if (tensor->buffer.defined() &&
        (utils::Endswith(tensor->buffer->name, "_read_cache") ||
         utils::Endswith(tensor->buffer->name, "_cache_write_out")) &&
        ((*global_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal || blockidx_)) {
      bool temp_replace = do_replace;
      do_replace        = true;
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
      do_replace = temp_replace;
    } else {
    }
  }

 private:
  std::map<std::string, ir::Tensor>* global_tensor_map_;
  std::unordered_set<std::string>* resized_buffer_;
  const Var& var_;
  const Expr& expr_;
  bool blockidx_;
  bool do_replace{false};
  bool find_replace{false};
  const Expr& extent_;
};

void CUDAReplaceIndexOfCachePass(Expr* source,
                                 const Var& var,
                                 const Expr& expr,
                                 std::map<std::string, ir::Tensor>* global_tensor_map,
                                 std::unordered_set<std::string>* resized_buffer,
                                 bool blockidx,
                                 const Expr& extent) {
  ReplaceVarIndexOfCacheMutator mutator(var, expr, global_tensor_map, resized_buffer, blockidx, extent);
  mutator.Execute(source);
}

}  // namespace optim
}  // namespace cinn

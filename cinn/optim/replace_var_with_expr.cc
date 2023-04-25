// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/optim/replace_var_with_expr.h"

#include "cinn/common/cas.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/tensor.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_const_param_to_integer.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprMutator : public ir::IRMutator<> {
  ReplaceVarWithExprMutator(const Var& var, const Expr& expr, const std::string& tensor_name)
      : var_(var), expr_(expr), tensor_name_(tensor_name) {}

  void operator()(Expr* expr) {
    if (tensor_name_.empty()) visit_all_ = true;
    IRMutator::Visit(expr, expr);
  }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name == var_->name && (do_replace_ || visit_all_)) {
      auto copied = IRCopy(expr_);
      *op         = copied;
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->min, &node->min);
    ir::IRMutator<>::Visit(&node->extent, &node->extent);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->loop_var->name == var_->name && expr_.As<ir::_Var_>() && visit_all_) {
      node->loop_var = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->init, &node->init);
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    ir::IRMutator<>::Visit(&node->inc, &node->inc);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->iterator->name == var_->name && expr_.As<ir::_Var_>() && visit_all_) {
      node->iterator = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node   = expr->As<ir::Store>();
    auto* tensor = node->tensor.as_tensor();

    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& index : node->indices) {
      ir::IRMutator<>::Visit(&index, &index);
    }
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
    ir::IRMutator<>::Visit(&node->value, &node->value);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node   = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }

 private:
  bool do_replace_{false};
  bool visit_all_{false};
  const Var& var_;
  const Expr& expr_;
  const std::string& tensor_name_;
};

void ReplaceVarWithExpr(Expr* source, const Var& var, const Expr& expr, const std::string& tensor_name) {
  ReplaceVarWithExprMutator mutator(var, expr, tensor_name);
  mutator(source);
}

struct CollectTensorIndexMutator : public ir::IRMutator<> {
  CollectTensorIndexMutator(const std::string& tensor_name) : tensor_name_(tensor_name) {}

  std::vector<std::vector<Expr>> operator()(Expr* expr) {
    IRMutator::Visit(expr, expr);
    return res;
  }

 private:
  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node   = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      res.push_back(node->indices);
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    }
  }

 private:
  std::vector<std::vector<Expr>> res;
  const std::string& tensor_name_;
};

std::vector<std::vector<Expr>> CollectTensorIndex(Expr* source, const std::string& tensor_name) {
  CollectTensorIndexMutator mutator(tensor_name);
  std::vector<std::vector<Expr>> result = mutator(source);
  for (auto& i : result) {
    for (auto& j : i) {
      j = common::AutoSimplify(j);
    }
  }
  return result;
}

struct ReplaceVarIndexOfCacheMutator : public ir::IRMutator<> {
  ReplaceVarIndexOfCacheMutator(const Var& var,
                                const Expr& expr,
                                std::map<std::string, ir::Tensor>* global_tensor_map,
                                std::unordered_map<std::string, std::vector<Expr>>& resized_buffer_cache,
                                bool blockidx,
                                const Expr& extent,
                                std::string tensor_name,
                                const std::map<std::string, int>& loop2extent)
      : var_(var),
        expr_(expr),
        global_tensor_map_(global_tensor_map),
        resized_buffer_cache_(resized_buffer_cache),
        blockidx_(blockidx),
        extent_(extent),
        tensor_name_(tensor_name),
        loop2extent_(loop2extent) {}

  void Execute(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void ResizeTempMemory(const std::string& tensor_name, int index, Expr* indice, const std::string& var_name) {
    if (extent_.defined() && extent_.is_constant()) {
      // To avoid duplicate resizing of the same buffer, we use a string of buffer name + var name + shape's index as
      // ID. If an ID already exists, which means we have already edited the buffer's shape, we will just return.
      std::string buffer_id =
          (*global_tensor_map_)[tensor_name]->buffer->name + "_" + var_->name + "_" + std::to_string(index);
      VLOG(2) << "ResizeTempMemory tensor_name [" << tensor_name << "], index [" << index << "], indice [" << *indice
              << "], var_name [" << var_name << "], buffer_id=" << buffer_id;
      auto cached_it = resized_buffer_cache_.find(buffer_id);
      if (cached_it != resized_buffer_cache_.end()) {
        std::vector<Expr> buffer_shape = IRCopy(cached_it->second);
        auto buffer_name               = (*global_tensor_map_).at(tensor_name)->buffer->name;
        VLOG(3) << buffer_id << " already resized! " << buffer_name << " 's shape=[" << utils::Join(buffer_shape, ",")
                << "]";
        (*global_tensor_map_).at(tensor_name)->shape         = buffer_shape;
        (*global_tensor_map_).at(tensor_name)->buffer->shape = IRCopy(buffer_shape);
        return;
      }
      auto buffer_shape              = IRCopy((*global_tensor_map_)[tensor_name]->buffer->shape);
      std::vector<Expr> tensor_shape = IRCopy((*global_tensor_map_).at(tensor_name)->shape);
      VLOG(3) << tensor_name << " tensor's Original Shape is : ";
      for (auto& i : tensor_shape) {
        VLOG(3) << i;
      }
      VLOG(3) << buffer_id << " buffer's Original Shape is : ";
      for (auto& i : buffer_shape) {
        VLOG(3) << i;
      }
      tensor_shape[index] = common::AutoSimplify(tensor_shape[index]);
      CHECK(tensor_shape[index].is_constant());
      CHECK(extent_.is_constant());

      // copy indice and replace var to 0
      auto indice_copy = IRCopy(*indice);
      ReplaceVarWithExpr(&indice_copy, Var(var_name), Expr(0));
      auto vars = ir::CollectIRNodesInOrder(indice_copy, [](const ir::Expr* expr) { return expr->As<ir::_Var_>(); });

      int max_range = 0;
      // using recursion funcitons index range.
      std::function<void(int, ir::Expr)> compute_range = [&](const int deep, ir::Expr index) {
        auto var = vars[deep].as_var_ref();
        CHECK(loop2extent_.count(var->name));
        auto extent = loop2extent_.find(var->name)->second;

        for (int idx = 0; idx < extent; ++idx) {
          auto tmp = IRCopy(index);
          ReplaceVarWithExpr(&tmp, var, Expr(idx));

          if (deep == vars.size() - 1) {
            auto simplify = common::AutoSimplify(tmp);
            auto range    = common::AutoSimplify(simplify);
            CHECK(range.is_constant());
            max_range = std::max(max_range, range.as_int32() + 1);
          } else {
            compute_range(deep + 1, tmp);
          }
        }
      };

      compute_range(0, indice_copy);
      tensor_shape[index] = Expr(max_range);

      (*global_tensor_map_).at(tensor_name)->shape      = tensor_shape;
      (*global_tensor_map_)[tensor_name]->buffer->shape = IRCopy(tensor_shape);
      resized_buffer_cache_.emplace(buffer_id, tensor_shape);
      VLOG(3) << tensor_name << " tensor and buffer " << (*global_tensor_map_).at(tensor_name)->buffer->name
              << "'s New Shape is : ";
      for (auto& i : (*global_tensor_map_).at(tensor_name)->shape) {
        VLOG(3) << i;
      }
      VLOG(3) << "Check buffer shape";
      for (auto& i : (*global_tensor_map_).at(tensor_name)->buffer->shape) {
        VLOG(3) << i;
      }
    } else {
      VLOG(3) << "extent not defined";
    }
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(node->schedule_block.As<ir::ScheduleBlock>());
    auto iter_values = node->iter_values;
    auto& body_copy  = node->schedule_block.As<ir::ScheduleBlock>()->body;
    auto iter_vars   = node->schedule_block.As<ir::ScheduleBlock>()->iter_vars;

    CHECK_EQ(iter_values.size(), iter_vars.size());
    for (int i = 0; i < iter_values.size(); i++) {
      ReplaceVarWithExpr(&body_copy, iter_vars[i], iter_values[i]);
    }

    for (auto& value : node->iter_values) {
      ir::IRMutator<>::Visit(&value, &value);
    }

    bool temp_find_replace = find_replace_;
    ir::IRMutator<>::Visit(&body_copy, &body_copy);
    find_replace_ = temp_find_replace;
  }

  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (do_replace_) {
      if (expr->name != utils::GetStreamCnt(var_->name)) return;
      VLOG(2) << "Do Replace: " << expr->name << " to " << expr_;
      auto copied   = IRCopy(expr_);
      *op           = copied;
      find_replace_ = true;
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node   = expr->As<ir::Store>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor_name_.empty() && tensor->buffer.defined() &&
        (utils::Endswith(tensor->buffer->name, "_read_cache") ||
         utils::Endswith(tensor->buffer->name, "_write_cache") ||
         utils::Endswith(tensor->buffer->name, "_temp_buffer")) &&
        ((*global_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal || blockidx_)) {
      VLOG(2) << "Store 's tensor name=" << tensor->name << ", buffer name=" << tensor->buffer->name;
      bool temp_replace = do_replace_;
      do_replace_       = true;
      find_replace_     = false;
      for (int i = 0; i < node->indices.size(); i++) {
        auto& temp = node->indices[i];
        VLOG(2) << "i=" << i << ",indice=" << temp;
        // When eliminating axis 'j_inner' in index '10 * j_outer + j_inner' (j_inner's extent is 10)
        // Divide '10 * j_outer' by 10, and get new index 'j_outer + j_inner'
        if (extent_.defined() && temp.As<ir::Add>() && temp.As<ir::Add>()->a().As<ir::Mul>() &&
            temp.As<ir::Add>()->b().as_var() && temp.As<ir::Add>()->b().as_var()->name == var_->name) {
          temp.As<ir::Add>()->a() = ir::Div::Make(temp.As<ir::Add>()->a(), extent_);
          VLOG(2) << "modify to indice=" << temp;
        }
        temp           = common::AutoSimplify(temp);
        auto temp_copy = IRCopy(temp);
        // Eliminate var 'j_inner' and get the final index 'j_outer'
        ir::IRMutator<>::Visit(&temp, &temp);
        if (find_replace_ == true) {
          VLOG(3) << "Find " << var_->name << " in indice: " << temp_copy;
          // If we replaced var 'j_inner'(the axis to be eliminated) to 0 in indices[i], edit tensor's shape[i] and
          // buffer's shape
          ResizeTempMemory(tensor->name, i, &temp_copy, var_->name);
          find_replace_ = false;
        }
      }

      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      do_replace_ = temp_replace;
      ir::IRMutator<>::Visit(&node->value, &node->value);
      for (int i = 0; i < node->indices.size(); i++) {
        Expr index = node->indices[i];
        index      = common::AutoSimplify(index);
        // If index[i] is 0, then edit the tensor's shape[i] to 1
        if (index.is_constant() && index.get_constant() == 0.0f) {
          std::vector<Expr> new_shape                   = (*global_tensor_map_).at(tensor->name)->shape;
          new_shape[i]                                  = Expr(1);
          (*global_tensor_map_).at(tensor->name)->shape = new_shape;
        }
      }
    } else if (tensor->buffer.defined() && !tensor_name_.empty() && tensor->buffer->name == tensor_name_) {
      bool temp_replace = do_replace_;
      do_replace_       = true;
      find_replace_     = false;
      for (int i = 0; i < node->indices.size(); i++) {
        auto& temp = node->indices[i];
        // When eliminating axis 'j_inner' in index '10 * j_outer + j_inner' (j_inner's extent is 10)
        // Divide '10 * j_outer' by 10, and get new index 'j_outer + j_inner'
        if (extent_.defined() && temp.As<ir::Add>() && temp.As<ir::Add>()->a().As<ir::Mul>() &&
            temp.As<ir::Add>()->b().as_var() && temp.As<ir::Add>()->b().as_var()->name == var_->name) {
          temp.As<ir::Add>()->a() = ir::Div::Make(temp.As<ir::Add>()->a(), extent_);
        }
        temp           = common::AutoSimplify(temp);
        auto temp_copy = IRCopy(temp);
        // Eliminate var 'j_inner' and get the final index 'j_outer'
        ir::IRMutator<>::Visit(&temp, &temp);
        if (find_replace_ == true) {
          VLOG(3) << "find " << var_->name << " in indice: " << temp_copy;
          // If we replaced var 'j_inner'(the axis to be eliminated) to 0 in indices[i], edit tensor's shape[i] and
          // buffer's shape
          ResizeTempMemory(tensor->name, i, &temp_copy, var_->name);
          find_replace_ = false;
        }
      }
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      do_replace_ = temp_replace;
      ir::IRMutator<>::Visit(&node->value, &node->value);
      for (int i = 0; i < node->indices.size(); i++) {
        Expr index = node->indices[i];
        index      = common::AutoSimplify(index);
        // If index[i] is 0, then edit the tensor's shape[i] to 1
        if (index.is_constant() && index.get_constant() == 0.0f) {
          std::vector<Expr> new_shape                   = (*global_tensor_map_).at(tensor->name)->shape;
          new_shape[i]                                  = Expr(1);
          (*global_tensor_map_).at(tensor->name)->shape = new_shape;
        }
      }
    } else {
      for (auto& index : node->indices) {
        ir::IRMutator<>::Visit(&index, &index);
      }
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      ir::IRMutator<>::Visit(&node->value, &node->value);
    }
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node   = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    VLOG(2) << "Load's tensor name is : " << tensor->name;
    if (tensor_name_.empty() && tensor->buffer.defined() &&
        (utils::Endswith(tensor->buffer->name, "_read_cache") ||
         utils::Endswith(tensor->buffer->name, "_write_cache") ||
         utils::Endswith(tensor->buffer->name, "_temp_buffer")) &&
        ((*global_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal || blockidx_)) {
      bool temp_replace = do_replace_;
      do_replace_       = true;
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
      do_replace_ = temp_replace;
    } else if (tensor->buffer.defined() && !tensor_name_.empty() && tensor->buffer->name == tensor_name_) {
      bool temp_replace = do_replace_;
      do_replace_       = true;
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
      do_replace_ = temp_replace;
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    }
  }

 private:
  std::map<std::string, ir::Tensor>* global_tensor_map_;
  std::unordered_map<std::string, std::vector<Expr>>& resized_buffer_cache_;

  const Var& var_;
  const Expr& expr_;
  bool blockidx_;
  bool do_replace_{false};
  bool find_replace_{false};
  const Expr& extent_;
  std::string tensor_name_;
  const std::map<std::string, int>& loop2extent_;
};

void CUDAReplaceIndexOfCachePass(Expr* source,
                                 const Var& var,
                                 const Expr& expr,
                                 std::map<std::string, ir::Tensor>* global_tensor_map,
                                 std::unordered_map<std::string, std::vector<Expr>>& resized_buffer_cache,
                                 bool blockidx,
                                 const Expr& extent,
                                 std::string tensor_name,
                                 const std::map<std::string, int>& loop2extent) {
  if (extent.defined() && !extent.is_constant()) {
    VLOG(3) << "Warning! The extent " << extent << " is not constant in CUDAReplaceIndexOfCachePass!";
  }
  VLOG(3) << "CUDAReplaceIndexOfCachePass with tensor_name [" << tensor_name << "] and extent [" << extent << "] visit "
          << *source;
  ReplaceVarIndexOfCacheMutator mutator(
      var, expr, global_tensor_map, resized_buffer_cache, blockidx, extent, tensor_name, loop2extent);
  mutator.Execute(source);
  VLOG(3) << "After replace, expr is : " << *source;
}

}  // namespace optim
}  // namespace cinn

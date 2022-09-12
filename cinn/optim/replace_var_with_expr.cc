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
      Simplify(&j);
    }
  }
  return result;
}

struct ReplaceVarIndexOfCacheMutator : public ir::IRMutator<> {
  ReplaceVarIndexOfCacheMutator(const Var& var,
                                const Expr& expr,
                                std::map<std::string, ir::Tensor>* global_tensor_map,
                                std::unordered_set<std::string>& resized_buffer,
                                bool blockidx,
                                const Expr& extent,
                                std::string tensor_name)
      : var_(var),
        expr_(expr),
        global_tensor_map_(global_tensor_map),
        resized_buffer_(resized_buffer),
        blockidx_(blockidx),
        extent_(extent),
        tensor_name_(tensor_name) {}

  void Execute(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void ResizeTempMemory(const std::string& tensor_name, int index, Expr* indice, const std::string& var_name) {
    if (extent_.defined() && extent_.is_constant()) {
      // To avoid duplicate resizing of the same buffer, we use a string of buffer name + var name + shape's index as
      // ID. If an ID already exists, which means we have already edited the buffer's shape, we will just return.
      VLOG(2) << "ResizeTempMemory tensor_name [" << tensor_name << "], index [" << index << "], indice [" << *indice
              << "], var_name [" << var_name << "].";
      std::string buffer_id =
          (*global_tensor_map_)[tensor_name]->buffer->name + "_" + var_->name + "_" + std::to_string(index);
      if (resized_buffer_.count(buffer_id) != 0) {
        auto buffer_name = (*global_tensor_map_).at(tensor_name)->buffer->name;
        VLOG(3) << buffer_id << " already resized! " << buffer_name << " 's shape is : ";
        std::vector<Expr> buffer_shape = IRCopy(buffer_shape_map_[buffer_name]);
        for (auto i : buffer_shape) VLOG(3) << i;
        (*global_tensor_map_).at(tensor_name)->shape         = buffer_shape;
        (*global_tensor_map_).at(tensor_name)->buffer->shape = IRCopy(buffer_shape);
        VLOG(3) << (*global_tensor_map_).at(tensor_name)->name << " 's shape is : ";
        for (auto i : (*global_tensor_map_).at(tensor_name)->shape) VLOG(3) << i;
        for (auto i : *global_tensor_map_) {
          if (i.second->buffer.defined() && i.second->buffer->name == buffer_name) {
            VLOG(3) << i.second->name << " 's buffer " << i.second->buffer->name << " shape is : ";
            for (auto j : i.second->buffer->shape) VLOG(3) << j;
          }
        }
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
      Simplify(&tensor_shape[index]);
      CHECK(tensor_shape[index].is_constant());
      CHECK(extent_.is_constant());
      /**
       * Here we need to calculate the new shape[index] after removing the var.
       * For example, if tensor A_temp's original shape is [100,100] and its indice is [i_outer * 10 + i_inner, j]. (0 <
       * i_outer < 10 and 0 < i_inner < 10 and 0 < j < 100) After removing the var i_outer, its new shape[0] should be:
       * diff = oldshape[0](when i_outer = 9) - oldshape[0](when i_outer = 0)
       * new_shape[0] = old_shape[0] - diff
       * In this case, new_shape[0] = 100 - 90 = 10
       * Thus we get A_temp's new shape: [10, 100] and new indice [i_inner, j]. (0 < i_inner < 10 and 0 < j < 100)
       */
      int extent_i = extent_.get_constant();
      auto copy1   = IRCopy(*indice);
      auto copy2   = IRCopy(*indice);
      ReplaceVarWithExpr(&copy1, Var(var_name), Expr(extent_i - 1));
      ReplaceVarWithExpr(&copy2, Var(var_name), Expr(0));
      auto res            = copy1 - copy2;
      tensor_shape[index] = tensor_shape[index] - res;
      Simplify(&tensor_shape[index]);
      VLOG(2) << "tensor_shape[index] - res is : " << tensor_shape[index];
      if (tensor_shape[index].is_constant() && tensor_shape[index].get_constant() <= 0) {
        tensor_shape[index] = Expr(1);
      } else if (!tensor_shape[index].is_constant()) {
        VLOG(3) << "Index is not constant: " << tensor_shape[index] << " and it will be replaced to 1";
        tensor_shape[index] = Expr(1);
      }
      (*global_tensor_map_).at(tensor_name)->shape = tensor_shape;

      resized_buffer_.insert(buffer_id);
      (*global_tensor_map_)[tensor_name]->buffer->shape                   = IRCopy(tensor_shape);
      buffer_shape_map_[(*global_tensor_map_)[tensor_name]->buffer->name] = tensor_shape;
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
    VLOG(2) << "Store 's tensor name is : " << tensor->name;
    if (tensor_name_.empty() && tensor->buffer.defined() &&
        (utils::Endswith(tensor->buffer->name, "_read_cache") ||
         utils::Endswith(tensor->buffer->name, "_write_cache") ||
         utils::Endswith(tensor->buffer->name, "_temp_buffer")) &&
        ((*global_tensor_map_).at(tensor->name)->buffer->memory_type == ir::MemoryType::GPULocal || blockidx_)) {
      bool temp_replace = do_replace_;
      do_replace_       = true;
      find_replace_     = false;
      VLOG(2) << tensor->name << " Store's indices size is : " << node->indices.size();
      for (int i = 0; i < node->indices.size(); i++) {
        auto& temp = node->indices[i];
        VLOG(2) << temp;
      }
      for (int i = 0; i < node->indices.size(); i++) {
        auto& temp = node->indices[i];
        // When eliminating axis 'j_inner' in index '10 * j_outer + j_inner' (j_inner's extent is 10)
        // Divide '10 * j_outer' by 10, and get new index 'j_outer + j_inner'
        if (extent_.defined() && temp.As<ir::Add>() && temp.As<ir::Add>()->a().As<ir::Mul>() &&
            temp.As<ir::Add>()->b().as_var() && temp.As<ir::Add>()->b().as_var()->name == var_->name) {
          temp.As<ir::Add>()->a() = ir::Div::Make(temp.As<ir::Add>()->a(), extent_);
        }
        Simplify(&temp);
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
        Simplify(&index);
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
        Simplify(&temp);
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
        Simplify(&index);
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
  std::map<std::string, std::vector<Expr>> buffer_shape_map_;
  std::unordered_set<std::string>& resized_buffer_;
  const Var& var_;
  const Expr& expr_;
  bool blockidx_;
  bool do_replace_{false};
  bool find_replace_{false};
  const Expr& extent_;
  std::string tensor_name_;
};

void CUDAReplaceIndexOfCachePass(Expr* source,
                                 const Var& var,
                                 const Expr& expr,
                                 std::map<std::string, ir::Tensor>* global_tensor_map,
                                 std::unordered_set<std::string>& resized_buffer,
                                 bool blockidx,
                                 const Expr& extent,
                                 std::string tensor_name) {
  if (extent.defined() && !extent.is_constant()) {
    VLOG(3) << "Warning! The extent " << extent << " is not constant in CUDAReplaceIndexOfCachePass!";
  }
  VLOG(3) << "CUDAReplaceIndexOfCachePass visit " << *source;
  ReplaceVarIndexOfCacheMutator mutator(var, expr, global_tensor_map, resized_buffer, blockidx, extent, tensor_name);
  mutator.Execute(source);
}

}  // namespace optim
}  // namespace cinn

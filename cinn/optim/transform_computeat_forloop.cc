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

#include "cinn/optim/transform_computeat_forloop.h"

#include <algorithm>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/stage.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace optim {

void MarkComputeAtForloop(const std::string &statement,
                          const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
                          std::map<std::string, ir::Tensor> *global_tensor_map,
                          std::unordered_set<std::string> &resized_buffer,
                          Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    const std::string &statement;
    const std::map<std::string, poly::StageForloopInfo> forloop_infos;
    std::map<std::string, ir::Tensor> *global_tensor_map;
    std::unordered_set<std::string> &resized_buffer;
    /**
     * @param statement the tuple name.
     * @param forloop_infos the axis.
     */
    Mutator(const std::string &statement,
            const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
            std::map<std::string, ir::Tensor> *global_tensor_map,
            std::unordered_set<std::string> &resized_buffer)
        : statement(statement),
          forloop_infos(forloop_infos),
          global_tensor_map(global_tensor_map),
          resized_buffer(resized_buffer) {}

    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    // Mark the specific store.
    void Visit(const ir::Store *op, Expr *expr) override {
      auto *tensor = op->tensor.As<ir::_Tensor_>();
      if (tensor->name == statement) {
        if (tensor->buffer.defined()) {
          MarkForloop(tensor->buffer->name);
        } else {
          MarkForloop("not_defined");
        }
      }
    }

    void MarkForloop(const std::string &tensor_name) {
      // start from 0, threadIdx.x
      for (auto *expr : forloop_stack) {
        VLOG(2) << "expr in forloop_stack is : \n" << *expr;
        auto *for_     = expr->As<ir::For>();
        auto *poly_for = expr->As<ir::PolyFor>();
        Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
        auto it        = forloop_infos.find(axis_var->name);
        VLOG(2) << "Poly_for/For iterator name is : " << axis_var->name;
        std::string iterator_name;
        if (it != forloop_infos.end()) {
          if (for_) {
            for_->set_for_type(it->second.for_type);
            for_->device_api = it->second.device;
            iterator_name    = for_->loop_var->name;
            VLOG(2) << "In this for loop, extent is : " << for_->extent;
            VLOG(2) << "In this for loop, body is : " << for_->body;
          } else {
            poly_for->set_for_type(it->second.for_type);
            poly_for->device_api = it->second.device;
            iterator_name        = poly_for->iterator->name;
            VLOG(2) << "In this poly_for loop, condition is : " << poly_for->condition;
            VLOG(2) << "In this poly_for loop, body is : " << poly_for->body;
          }
          if (it->second.for_type == ir::ForType::Default) {
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(2) << "ComputeAt replacing var " << axis_var->name << " to Expr(0) in tensor " << tensor_name;
            optim::CUDAReplaceIndexOfCachePass(
                expr, axis_var, ir::Expr(0), global_tensor_map, resized_buffer, false, extent, tensor_name);
          } else {
            CINN_NOT_IMPLEMENTED
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) override {
      forloop_stack.push_back(expr);
      IRMutator::Visit(op, expr);
      forloop_stack.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) override {
      forloop_stack.push_back(expr);
      IRMutator::Visit(op, expr);
      forloop_stack.pop_back();
    }

    std::vector<Expr *> forloop_stack;
  };

  Mutator mutator(statement, forloop_infos, global_tensor_map, resized_buffer);
  mutator(expr);
}

void TransformComputeatForloops(const forloop_infos_t &forloop_infos,
                                const std::vector<std::string> &traverse_order,
                                std::map<std::string, ir::Tensor> *global_tensor_map,
                                std::unordered_set<std::string> &resized_buffer,
                                Expr *expr) {
  for (auto &i : traverse_order) {
    MarkComputeAtForloop(i, forloop_infos.at(i), global_tensor_map, resized_buffer, expr);
  }
}

}  // namespace optim
}  // namespace cinn

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

#include "cinn/optim/unroll_loops.h"

#include <vector>

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_replace.h"

namespace cinn {
namespace optim {

namespace {

struct UnrollMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::For* op, Expr* expr) override {
    /*     LOG(INFO) << "Check loop var [" << op->loop_var <<"]";
        auto f_type = op->for_type();
        LOG(INFO) << "op->ForType is : " << *reinterpret_cast<int*>(&f_type);
        if (op->is_unrolled()) LOG(INFO)<< "It is unrolled";
        if (op->is_serial()) LOG(INFO)<< "It is_serial";
        if (op->is_vectorized()) LOG(INFO)<< "It is_vectorized";
        if (op->is_parallel()) LOG(INFO)<< "It is_parallel";
        LOG(INFO) << "Its extent is : [" << op->extent << "]";
        if (op->is_unrolled() && op->extent.is_constant()) LOG(INFO)<< "Its extent is constant";
        if (op->is_unrolled() && op->extent.as_int32()) LOG(INFO)<< "Its extent is int32"; */
    if (is_unrollable(op)) {
      // LOG(INFO) << "Unroll loop var [" << op->loop_var<<"]";
      Unroll(op, expr);
      IRMutator<>::Visit(expr, expr);
    } else {
      auto* node = expr->As<ir::For>();
      ir::IRMutator<>::Visit(&node->body, &node->body);
    }
  }

  bool is_unrollable(const ir::For* op) const {
    return op->is_unrolled() && op->extent.is_constant() && op->extent.as_int32() < 50;
  }

  //! Unroll a forloop.
  void Unroll(const ir::For* op, Expr* expr) {
    std::vector<Expr> body;

    auto* min    = op->min.As<ir::IntImm>();
    auto* extent = op->extent.As<ir::IntImm>();
    if (!(min && extent)) return;

    for (int i = min->value; i < extent->value; i++) {
      Expr start = op->min + i;
      body.push_back(optim::IRCopy(op->body));
      optim::IrReplace(&body.back(), op->loop_var, start);
    }

    *expr = ir::Block::Make(body);
  }
};

}  // namespace

void UnrollLoop(Expr* expr) {
  // LOG(INFO) << "UnrollLoop: " << *expr;
  UnrollMutator()(expr);
}

}  // namespace optim
}  // namespace cinn

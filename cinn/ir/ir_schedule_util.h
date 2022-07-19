// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/tensor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
};

// Self-defined operator to support std::set<Var>
struct CompVar {
  bool operator()(const Var& left, const Var& right) const { return left->name < right->name; }
};

struct MappingVarToExprMutator : public ir::IRMutator<> {
  MappingVarToExprMutator(const std::map<Var, Expr, CompVar>& replacing_map) : replacing_map_(replacing_map) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (replacing_map_.count(op->as_var_ref())) {
      *op = replacing_map_.at(op->as_var_ref());
    }
  }

 private:
  const std::map<Var, Expr, CompVar>& replacing_map_;
};

struct CacheBlockInfo {
  /*! \brief The tensor to be read. */
  Tensor read_tensor;
  /*! \brief The tensor to be written. */
  Tensor write_tensor;
  /*! \brief The tensor allocation to be inserted into the block signature. */
  Tensor alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  Expr loc_block;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Expr cache_block;
};

Tensor GetTensor(const Expr& block);

Tensor GetReadTensor(const Expr& block, int index);

int GetLoopExtent(const Expr& loop);

bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name);

void SetCudaAxisInfo(Expr* lowered_func);

bool Contains(const Expr& container, const Expr& expr);

Expr GetNextForLoop(const Expr& for_loop);

std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom);

void ReplaceExpr(Expr* source, const std::vector<Var>& replaced, const std::vector<Expr>& candidates);

std::vector<int> ValidateFactors(const std::vector<int>& factors, int total_extent);

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis);

std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root);

std::pair<Expr, Expr> GetRange(Expr index,
                               const std::vector<Var>& iter_vars,
                               const std::vector<std::pair<Expr, Expr>>& iter_range,
                               int i);

std::vector<std::pair<Expr, Expr>> GetRange(const std::vector<Expr>& tensor_indices,
                                            const std::vector<Var>& iter_vars,
                                            const std::vector<std::pair<Expr, Expr>>& iter_range,
                                            const Tensor& tensor);

std::vector<std::pair<Expr, Expr>> CalculateTensorRegions(const Expr& block,
                                                          const std::vector<Expr>& tensor_indices,
                                                          const Tensor& tensor,
                                                          const Expr& root);

Expr GetNthAccessExpr(const Expr& block, int index, bool is_write);

Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type);

Expr MakeCacheBlock(const std::vector<std::pair<Expr, Expr>>& buffer_region,
                    CacheBlockInfo* info,
                    const std::string& memory_type,
                    DeviceAPI device_api);

void FindInsertionPoint(Expr& root, CacheBlockInfo* info, bool is_write);

const std::set<Expr, CompExpr> CollectLoopsToSet(const std::vector<Expr>& loops);

std::pair<Expr, Expr> GetBoundaryOfReorderRange(const std::set<Expr, CompExpr>& loop_set);

std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom);

Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes);

std::vector<Expr> GetProducers(const Expr& block, const Expr& root);

std::vector<Expr> GetConsumers(const Expr& block, const Expr& root);

void CheckComputeAtValidation(const Expr& block, const Expr& loop, const Expr& root);

void InsertBlock(Expr& for_loop, const Expr& insertion);

std::pair<Expr, Expr> RangeUnion(const std::pair<Expr, Expr>& range1, const std::pair<Expr, Expr>& range2);

std::vector<std::pair<Expr, Expr>> CalculateRequiredRegions(const Expr& block,
                                                            const Expr& loop,
                                                            const std::vector<Expr>& consumers,
                                                            const Expr& root);

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block, const Expr& root);

}  // namespace ir
}  // namespace cinn

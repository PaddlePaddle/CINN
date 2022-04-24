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
#include <iostream>
#include <vector>

#include "cinn/common/cinn_value.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace op {
namespace util {

framework::shape_t GetShape(const ir::Tensor& x);

framework::shape_t CheckAndValidReduceDim(const framework::shape_t& dim, const size_t rank);

enum class ReduceFuncType : int {
  Reduce = 0,
  BlockReduce,
  BlockReduceInternal,
  ReduceWithInternal,
  ReduceForSmallerDim
};

std::string ReduceFuncType2String(const ReduceFuncType& reduce_func_type);

ReduceFuncType SelectReduceFuncType(const std::vector<ir::Expr>& input_shape,
                                    const framework::shape_t& dim,
                                    const common::Target& target);

std::vector<ir::Tensor> RunReduceCompute(const ReduceFuncType& reduce_func_type,
                                         const std::string& op_name,
                                         const ir::Tensor& x,
                                         const framework::shape_t& dim,
                                         bool keep_dim                  = false,
                                         const std::string& output_name = UniqName("T_reduce_out"));

void RunReduceSchedule(const ReduceFuncType& reduce_func_type,
                       const std::vector<ir::Expr>& input_shape,
                       const framework::shape_t& dim,
                       const common::Target& target,
                       common::CINNValuePack* arg_pack);

}  // namespace util
}  // namespace op
}  // namespace hlir
}  // namespace cinn

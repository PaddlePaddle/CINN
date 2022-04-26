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

#include "cinn/hlir/op/reduction_util.h"

#include <iostream>
#include <vector>

#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/reduction.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace op {
namespace util {

using framework::dim_t;
using framework::shape_t;
using ir::Expr;

using ReduceComputeFunc =
    std::function<ir::Tensor(const ir::Tensor&, const shape_t&, bool, ir::Expr, const std::string&)>;
using BlockReduceInternalComputeFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor&, const int, const bool, const std::string&)>;
using BlockReduceComputeFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor&, const int, const int, const bool, const std::string&)>;
using ReduceForSmallerDimComputeFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor&, const shape_t&, bool, const std::string&)>;

using RunReduceComputeFunc = std::function<std::vector<ir::Tensor>(
    const std::string&, const ir::Tensor&, const shape_t&, bool, const std::string&)>;
using RunReduceScheduleFunc =
    std::function<void(const shape_t&, const shape_t&, const common::Target&, common::CINNValuePack*)>;

std::string ReduceFuncType2String(const ReduceFuncType& reduce_func_type) {
  static std::unordered_map<ReduceFuncType, std::string> run_reduce_func_map = {
      {ReduceFuncType::Reduce, "Reduce"},
      {ReduceFuncType::BlockReduce, "BlockReduce"},
      {ReduceFuncType::BlockReduceInternal, "BlockReduceInternal"},
      {ReduceFuncType::ReduceWithInternal, "ReduceWithInternal"},
      {ReduceFuncType::ReduceForSmallerDim, "ReduceForSmallerDim"}};

  CHECK(run_reduce_func_map.count(reduce_func_type))
      << "Do not support ReduceFuncType " << static_cast<int>(reduce_func_type) << " ! Please check.";

  return run_reduce_func_map.at(reduce_func_type);
}

shape_t GetShape(const ir::Tensor& x) {
  auto last_reduce_dim = x->shape[2].as_int32() * x->shape[3].as_int32();
  // Split into last_reduce_dim into {n,k}
  shape_t new_shape = {x->shape[0].as_int32(), x->shape[1].as_int32()};
  // As the max block size is 1024, setting 1024 as limit
  if (last_reduce_dim <= 1024) {
    new_shape.push_back(last_reduce_dim);
  } else {
    // As sum of reduce dimension is over 1024, so find a value along(1024, 1) that can be divied by
    // last_reduce_dim.
    for (int idx = 1024;; --idx) {
      if (last_reduce_dim % idx == 0) {
        new_shape.push_back(last_reduce_dim / idx);
        new_shape.push_back(idx);
        break;
      }
    }

    CHECK_EQ(new_shape.size(), 4) << "Can't find a new shape that satisfy the requirement!";
  }

  return new_shape;
}

shape_t CheckAndValidReduceDim(const shape_t& dim, const size_t rank) {
  auto new_dim = dim;

  if (new_dim.empty()) {
    for (int i = 0; i < rank; ++i) {
      new_dim.push_back(i);
    }
  } else {
    // support dim[i] < 0 like as dim=[-1]
    for (int i = 0; i < new_dim.size(); ++i) {
      if (new_dim[i] < 0) {
        new_dim[i] += rank;
      }
      CHECK(new_dim[i] >= 0 && new_dim[i] < rank) << "The value of [dim] of Reduce should between 0 and " << rank
                                                  << ", but here " << dim[i] << " at " << i << " ! Please check.";
    }
  }

  std::sort(new_dim.begin(), new_dim.end());
  // check dim
  CHECK_LE(new_dim.size(), rank);
  for (int idx = 1; idx < new_dim.size(); ++idx) {
    CHECK_NE(new_dim[idx - 1], new_dim[idx]);
  }

  return new_dim;
}

ReduceFuncType SelectReduceWithLastDimFuncType(const shape_t& dim, const shape_t& input_shape) {
  // compute reduce args
  bool reduce_dim_succesive = true;
  dim_t last_succesive_dim  = input_shape.back();
  for (int idx = dim.size() - 2; idx >= 0; --idx) {
    if (dim[idx] != dim[idx + 1] - 1) {
      reduce_dim_succesive = false;
      break;
    } else {
      if (last_succesive_dim * input_shape[dim[idx]] > 1024) {
        reduce_dim_succesive = false;
        break;
      }
      last_succesive_dim *= input_shape[dim[idx]];
    }
  }

  if (reduce_dim_succesive) {          // the reduce dimension is succesive
    if (last_succesive_dim <= 1024) {  // if the succesive reduce dimension size <= 1024
      return ReduceFuncType::BlockReduceInternal;
    } else {  // if the succesive reduce dimension size > 256
      return ReduceFuncType::BlockReduce;
    }
  }
  // the reduce dimension is not succesive
  return ReduceFuncType::ReduceWithInternal;
}

ReduceFuncType SelectReduceWithOutLastDimFuncType(const shape_t& dim, const shape_t& input_shape) {
  // compute reduce dim
  dim_t last_dim = 1;
  for (int i = dim.back() + 1; i < input_shape.size(); ++i) {
    last_dim *= input_shape[i];
  }

  if (last_dim <= 128) {
    return ReduceFuncType::ReduceForSmallerDim;
  }
  return ReduceFuncType::Reduce;
}

ReduceFuncType SelectReduceFuncType(const std::vector<ir::Expr>& input_shape,
                                    const shape_t& dim,
                                    const common::Target& target) {
  if (target != common::DefaultNVGPUTarget()) {
    return ReduceFuncType::Reduce;
  }
  auto input_shape_int = ToShapeType(input_shape);
  return dim.back() == input_shape.size() - 1 ? SelectReduceWithLastDimFuncType(dim, input_shape_int)
                                              : SelectReduceWithOutLastDimFuncType(dim, input_shape_int);
}

std::vector<ir::Tensor> RunReduceBaseCompute(const std::string& op_name,
                                             const ir::Tensor& x,
                                             const shape_t& dim,
                                             bool keep_dim                  = false,
                                             const std::string& output_name = "T_Reduce_out") {
  static std::unordered_map<std::string, ReduceComputeFunc> reduce_func_map = {{"reduce_sum", pe::ReduceSum},
                                                                               {"reduce_prod", pe::ReduceProd},
                                                                               {"reduce_max", pe::ReduceMax},
                                                                               {"reduce_min", pe::ReduceMin}};

  CHECK(reduce_func_map.count(op_name)) << "RunReduceBaseCompute Not support op [" << op_name << "] ! Please check.";

  VLOG(3) << "Compute [" << op_name << "] by ReduceComputeFunc !";
  return std::vector<ir::Tensor>{reduce_func_map.at(op_name)(x, dim, keep_dim, ir::Expr(), UniqName(output_name))};
}

std::vector<ir::Tensor> RunBlockReduceCompute(const std::string& op_name,
                                              const ir::Tensor& x,
                                              const shape_t& dim,
                                              bool keep_dim                  = false,
                                              const std::string& output_name = "T_Reduce_out") {
  static std::unordered_map<std::string, BlockReduceComputeFunc> reduce_func_map = {
      {"reduce_sum", pe::BlockReduceSum},
      {"reduce_prod", pe::BlockReduceProd},
      {"reduce_max", pe::BlockReduceMax},
      {"reduce_min", pe::BlockReduceMin}};

  CHECK(reduce_func_map.count(op_name)) << "RunBlockReduceCompute Not support op [" << op_name << "] ! Please check.";

  VLOG(3) << "Do BlockReduce Compute!";
  int block_size = 1024;
  return reduce_func_map.at(op_name)(x, static_cast<int>(dim.size()), block_size, keep_dim, UniqName(output_name));
}

std::vector<ir::Tensor> RunBlockReduceInternalCompute(const std::string& op_name,
                                                      const ir::Tensor& x,
                                                      const shape_t& dim,
                                                      bool keep_dim                  = false,
                                                      const std::string& output_name = "T_Reduce_out") {
  static std::unordered_map<std::string, BlockReduceInternalComputeFunc> reduce_func_map = {
      {"reduce_sum", pe::BlockReduceSumInternal},
      {"reduce_prod", pe::BlockReduceProdInternal},
      {"reduce_max", pe::BlockReduceMaxInternal},
      {"reduce_min", pe::BlockReduceMinInternal}};

  CHECK(reduce_func_map.count(op_name)) << "RunBlockReduceInternalCompute Not support op [" << op_name
                                        << "] ! Please check.";

  VLOG(3) << "Do BlockReduceInternal Compute!";
  return reduce_func_map.at(op_name)(x, static_cast<int>(dim.size()), keep_dim, UniqName(output_name));
}

std::vector<ir::Tensor> RunReduceWithInternalCompute(const std::string& op_name,
                                                     const ir::Tensor& x,
                                                     const shape_t& dim,
                                                     bool keep_dim                  = false,
                                                     const std::string& output_name = "T_Reduce_out") {
  VLOG(3) << "Do Reduce And BlockReduceInternal Compute!";

  // compute reduce args
  auto input_shape         = ToShapeType(x->shape);
  int succesive_dim_idx    = 0;
  dim_t last_succesive_dim = input_shape.back();
  for (int idx = dim.size() - 2; idx >= 0; --idx) {
    if (dim[idx] != dim[idx + 1] - 1) {
      succesive_dim_idx = idx + 1;
      break;
    } else {
      if (last_succesive_dim * input_shape[dim[idx]] > 1024) {
        succesive_dim_idx = idx + 1;
        break;
      }
      last_succesive_dim *= input_shape[dim[idx]];
    }
  }

  // compute the parallel reduce dimension size
  shape_t reduce_without_last_diemension(dim.begin(), dim.begin() + succesive_dim_idx);
  // TODO(sunli) : support last dimension size over 1024
  CHECK_LE(last_succesive_dim, 1024) << "last dimension size is over 1024";

  // first: do reduce without last dimension
  auto reduce_out =
      RunReduceBaseCompute(op_name, x, reduce_without_last_diemension, keep_dim, UniqName(output_name + "_tmp"));

  // second: do reduce on last dimension
  shape_t reduce_last_diemension(dim.begin() + succesive_dim_idx, dim.end());
  auto out =
      RunBlockReduceInternalCompute(op_name, reduce_out[0], reduce_last_diemension, keep_dim, UniqName(output_name));

  out.insert(out.end(), reduce_out.begin(), reduce_out.end());
  return out;
}

std::vector<ir::Tensor> RunReduceForSmallerDimCompute(const std::string& op_name,
                                                      const ir::Tensor& x,
                                                      const shape_t& dim,
                                                      bool keep_dim                  = false,
                                                      const std::string& output_name = "T_Reduce_out") {
  static std::unordered_map<std::string, ReduceForSmallerDimComputeFunc> reduce_func_map = {
      {"reduce_sum", pe::ReduceSumForSmallerDim},
      {"reduce_prod", pe::ReduceProdForSmallerDim},
      {"reduce_max", pe::ReduceMaxForSmallerDim},
      {"reduce_min", pe::ReduceMinForSmallerDim}};

  CHECK(reduce_func_map.count(op_name)) << "RunReduceForSmallerDimCompute Not support op [" << op_name
                                        << "] ! Please check.";

  VLOG(3) << "Do ReduceForSmallerDim Compute!";
  return reduce_func_map.at(op_name)(x, dim, keep_dim, UniqName(output_name));
}

std::vector<ir::Tensor> RunReduceCompute(const ReduceFuncType& reduce_func_type,
                                         const std::string& op_name,
                                         const ir::Tensor& x,
                                         const shape_t& dim,
                                         bool keep_dim,
                                         const std::string& output_name) {
  static std::unordered_map<ReduceFuncType, RunReduceComputeFunc> run_reduce_func_map = {
      {ReduceFuncType::Reduce, RunReduceBaseCompute},
      {ReduceFuncType::BlockReduce, RunBlockReduceCompute},
      {ReduceFuncType::BlockReduceInternal, RunBlockReduceInternalCompute},
      {ReduceFuncType::ReduceWithInternal, RunReduceWithInternalCompute},
      {ReduceFuncType::ReduceForSmallerDim, RunReduceForSmallerDimCompute}};

  CHECK(run_reduce_func_map.count(reduce_func_type))
      << "Do not support ReduceFuncType " << static_cast<int>(reduce_func_type) << " ! Please check.";

  return run_reduce_func_map.at(reduce_func_type)(op_name, x, dim, keep_dim, UniqName(output_name));
}

void RunBlockReduceInternalSchedule(const shape_t& input_shape,
                                    const shape_t& dim,
                                    const common::Target& target,
                                    common::CINNValuePack* arg_pack) {
  CHECK_EQ(arg_pack->size(), 3UL);
  Expr out              = (*arg_pack)[0];
  Expr tmp_out          = (*arg_pack)[1];
  poly::StageMap stages = arg_pack->back();

  VLOG(3) << "Do CudaScheduleBlockReduceInternal Schedule!";
  pe::CudaScheduleBlockReduceInternal(stages, tmp_out.as_tensor_ref(), out.as_tensor_ref(), target);
}

void RunBlockReduceSchedule(const shape_t& input_shape,
                            const shape_t& dim,
                            const common::Target& target,
                            common::CINNValuePack* arg_pack) {
  CHECK_EQ(arg_pack->size(), 4UL);
  Expr out              = (*arg_pack)[0];
  Expr tmp_out          = (*arg_pack)[1];
  Expr reduce_tmp_out   = (*arg_pack)[2];
  poly::StageMap stages = arg_pack->back();

  VLOG(3) << "Do CudaScheduleBlockReduce Schedule!";
  pe::CudaScheduleBlockReduce(
      stages, reduce_tmp_out.as_tensor_ref(), tmp_out.as_tensor_ref(), out.as_tensor_ref(), target);
}

void RunReduceForSmallerDimSchedule(const shape_t& input_shape,
                                    const shape_t& dim,
                                    const common::Target& target,
                                    common::CINNValuePack* arg_pack) {
  CHECK_EQ(arg_pack->size(), 3UL);
  Expr out              = (*arg_pack)[0];
  Expr tmp_out          = (*arg_pack)[1];
  poly::StageMap stages = arg_pack->back();
  VLOG(3) << "Do CudaScheduleReduceForSmallerDim Schedule!";
  pe::CudaScheduleReduceForSmallerDim(stages, dim, tmp_out.as_tensor_ref(), out.as_tensor_ref());
}

void RunReduceBaseSchedule(const shape_t& input_shape,
                           const shape_t& dim,
                           const common::Target& target,
                           common::CINNValuePack* arg_pack) {
  CHECK_EQ(arg_pack->size(), 2UL);
  Expr out              = (*arg_pack)[0];
  poly::StageMap stages = arg_pack->back();
  VLOG(3) << "Do CudaScheduleReduceBase Schedule!";
  pe::CudaScheduleReduce(stages, out.as_tensor_ref(), input_shape.size() - dim.back() - 1, target);
}

void RunReduceSchedule(const ReduceFuncType& reduce_func_type,
                       const std::vector<ir::Expr>& input_shape,
                       const shape_t& dim,
                       const common::Target& target,
                       common::CINNValuePack* arg_pack) {
  static std::unordered_map<ReduceFuncType, RunReduceScheduleFunc> run_reduce_func_map = {
      {ReduceFuncType::Reduce, RunReduceBaseSchedule},
      {ReduceFuncType::BlockReduce, RunBlockReduceInternalSchedule},
      {ReduceFuncType::BlockReduceInternal, RunBlockReduceInternalSchedule},
      {ReduceFuncType::ReduceWithInternal, RunBlockReduceSchedule},
      {ReduceFuncType::ReduceForSmallerDim, RunReduceForSmallerDimSchedule}};

  CHECK(run_reduce_func_map.count(reduce_func_type))
      << "Do not support ReduceFuncType " << static_cast<int>(reduce_func_type) << " ! Please check.";

  return run_reduce_func_map.at(reduce_func_type)(ToShapeType(input_shape), dim, target, arg_pack);
}

}  // namespace util
}  // namespace op
}  // namespace hlir
}  // namespace cinn
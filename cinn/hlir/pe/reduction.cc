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

#include "cinn/hlir/pe/reduction.h"

#include <cinn/ir/ir_base.h>

#include <algorithm>

#include "cinn/common/ir_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using ir::Tensor;
using lang::Compute;

/**
 * @brief transform reduction axes which could be empty or have negative elements into real axes with valid dimension
 * indices.
 *
 * @param ndim Number of dimensions of the output tensor.
 * @param axes The axes parameter.
 * @param real_axes A non-empty sorted array of valid dimension indices, with no duplicates.
 *
 * @notes If the input axes are empty, the result will be axes including all dimensions. If any input element is
 * negative, it will be treated as an offset from the last dimension (same as python indexing rules).
 */
void GetRealAxes(int ndim, const std::vector<int>& axes, std::vector<int>* real_axes) {
  CHECK(real_axes);
  if (axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      real_axes->push_back(i);
    }
  } else {
    for (auto axis : axes) {
      if (axis < 0) {
        axis += ndim;
      }
      CHECK_LE(axis, ndim) << "exceeds the maximum dimension: " << ndim << std::endl;
      CHECK_GE(axis, 0);
      real_axes->push_back(axis);
    }
    real_axes->resize(std::unique(real_axes->begin(), real_axes->end()) - real_axes->begin());
    std::sort(real_axes->begin(), real_axes->end());
  }
}

/**
 * @brief Calculate the target reduced shape.
 *
 * @param real_axes A non-empty sorted array of valid dimension indices, with no duplicates.
 * @param output_shape The output Tensor shape.
 * @param tensor The input tensor.
 * @param keep_dims If this is set to true, the reduced axes are kept as dimensions with size one. This enables the
 * result to broadcast correctly against the input array.
 */
void GetOutputShape(const std::vector<int>& real_axes,
                    std::vector<Expr>* output_shape,
                    const Tensor& tensor,
                    bool keep_dims) {
  CHECK(output_shape);
  auto ndim = tensor->shape.size();
  if (keep_dims) {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axes.begin(), real_axes.end(), i) != real_axes.end()) {
        output_shape->push_back(common::make_one());
      } else {
        output_shape->push_back(tensor->shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axes.begin(), real_axes.end(), i) == real_axes.end()) {
        output_shape->push_back(tensor->shape[i]);
      }
    }
  }
  if (output_shape->empty()) {
    output_shape->push_back(common::make_one());
  }
}

/*!
 * @brief Create a reduction PE.
 *
 * @param tensor The input tensor.
 * @param fn The reduction function eg. ReduceSum
 * @param output_shape The output Tensor shape.
 * @param real_axes The real axes where the reduction is performed.
 * @param squeeze_axes The real axes to squeeze. If unsqueezed, reduced axes will have shape 1 in the output tensor.
 * @param initial Starting value for the sum.
 * @param output_name The name of the output Tensor.
 *
 * @return The result tensor.
 */
template <typename FuncOp>
Tensor DoReduce(const Tensor& tensor,
                const FuncOp& fn,
                const std::vector<Expr>& output_shape,
                const std::vector<int>& real_axes,
                const std::vector<int>& squeeze_axes,
                Expr initial,
                const std::string& output_name) {
  std::vector<Var> reduce_axes;
  for (auto& axis : real_axes) {
    std::string name = UniqName("kk");
    reduce_axes.push_back(Var(tensor->shape[axis], name));
  }
  auto compute = [&](const std::vector<Expr>& indices) -> Expr {
    std::vector<Expr> eval_indice;
    int indice_cnt = 0;
    int reduce_cnt = 0;

    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      bool squeeze_i = std::find(squeeze_axes.begin(), squeeze_axes.end(), i) != squeeze_axes.end();
      if (std::find(real_axes.begin(), real_axes.end(), i) != real_axes.end()) {
        eval_indice.push_back(reduce_axes[reduce_cnt]);
        reduce_cnt++;
        indice_cnt += !squeeze_i;
        continue;
      }
      eval_indice.push_back(indices[indice_cnt]);
      indice_cnt++;
    }
    return fn(tensor(eval_indice), reduce_axes, initial);
  };

  Tensor C = Compute(output_shape, compute, output_name);
  return C;
}

/**
 * @brief reduction PE
 *
 * @param tensor The input tensor.
 * @param axes The axes along which the reduction are performed.
 * @param fn The reduction function eg. ReduceSum
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * @param initial Starting value for the sum.
 *
 * @return The result tensor.
 */
template <typename FuncOp>
Tensor Reduce(const Tensor& tensor,
              const std::vector<int>& axes,
              const FuncOp& fn,
              bool keep_dims,
              ir::Expr initial,
              const std::string& output_name) {
  auto ndim = tensor->shape.size();
  CHECK_GT(ndim, 0) << "Reduce tensor's dim must be more than 0";
  std::vector<int> real_axes;
  GetRealAxes(static_cast<int>(ndim), axes, &real_axes);
  std::vector<Expr> output_shapes;
  GetOutputShape(real_axes, &output_shapes, tensor, keep_dims);
  return DoReduce(
      tensor, fn, output_shapes, real_axes, keep_dims ? std::vector<int>() : real_axes, initial, output_name);
}

Tensor ReduceSum(const Tensor& A, const std::vector<int>& axes, const bool keep_dims, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceSum, keep_dims, ir::Expr(0.0f), output_name);
}

Tensor ReduceProd(const Tensor& A, const std::vector<int>& axes, const bool keep_dims, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceMul, keep_dims, ir::Expr(1.0f), output_name);
}

Tensor ReduceMax(const Tensor& A, const std::vector<int>& axes, const bool keep_dims, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceMax, keep_dims, ir::Expr(-3.402823e+38f), output_name);
}

Tensor ReduceMin(const Tensor& A, const std::vector<int>& axes, const bool keep_dims, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceMin, keep_dims, Expr(3.402823e+38f), output_name);
}

std::vector<Tensor> WarpReduce(const ir::Tensor& A,
                               const int last_reduce_dim_num,
                               const bool keep_dim,
                               const std::string& reduce_type,
                               const std::string& output_name) {
  // compute shape size without last reduce dimension.
  int shape_size_without_reduce_dim = A->shape.size() - last_reduce_dim_num;

  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = shape_size_without_reduce_dim; idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // comput tmp output shape.
  std::vector<Expr> tmp_shape(A->shape.begin(), A->shape.begin() + shape_size_without_reduce_dim);
  tmp_shape.push_back(Expr(32));
  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + indexs.size() - 1);
        for (int idx = 0; idx < last_reduce_dim_num; ++idx) {
          tmp_indexs.push_back(Expr(0));
        }
        CHECK_EQ(A->shape.size(), tmp_indexs.size());
        Expr offset = common::IndiceToAbsOffset(A->shape, tmp_indexs);
        return lang::CallExtern(reduce_type, {A, offset, reduce_width});
      },
      UniqName(output_name + "_" + reduce_type));

  // compute ouput shape.
  std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + shape_size_without_reduce_dim);
  for (int idx = 0; idx < last_reduce_dim_num && keep_dim; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + shape_size_without_reduce_dim);
        tmp_indexs.push_back(Expr(0));
        return tmp_out(tmp_indexs);
      },
      UniqName(output_name));

  return {out, tmp_out};
}

std::vector<ir::Tensor> WarpReduceMax(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A, last_reduce_dim_num, keep_dim, "cinn_warp_reduce_max", output_name);
}

std::vector<ir::Tensor> WarpReduceSum(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A, last_reduce_dim_num, keep_dim, "cinn_warp_reduce_sum", output_name);
}

std::vector<ir::Tensor> WarpReduceAvg(const ir::Tensor& A,
                                      const int last_reduce_dim_num,
                                      const bool keep_dim,
                                      const std::string& output_name) {
  return WarpReduce(A, last_reduce_dim_num, keep_dim, "cinn_warp_reduce_avg", output_name);
}

std::vector<ir::Tensor> BlockReduceInternal(const ir::Tensor& A,
                                            const std::vector<int>& axes,
                                            const bool keep_dim,
                                            const std::string& reduce_type,
                                            const std::string& output_name) {
  CHECK_GE(A->shape.size(), axes.back() + 1) << "Axes is over size!";
  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = axes.front(); idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // compute tmp output shape.
  std::vector<Expr> tmp_shape(A->shape.begin(), A->shape.begin() + axes.front());
  tmp_shape.push_back(reduce_width);

  // compute the reduce dimension stride.
  std::vector<Expr> last_reduce_stride(A->shape.size() - axes.front(), Expr(1));
  for (int idx = A->shape.size(), index = int(last_reduce_stride.size()) - 2; index >= 0; --index) {
    last_reduce_stride[index] = last_reduce_stride[index + 1] * A->shape[--idx];
  }

  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        // comput index map from output to input.
        auto last_index = indexs.back();
        std::vector<Expr> input_indexs(indexs.begin(), indexs.begin() + indexs.size() - 1);
        for (int idx = 0; idx < A->shape.size() - axes.front(); ++idx) {
          input_indexs.push_back(last_index / last_reduce_stride[idx]);
          last_index = last_index % last_reduce_stride[idx];
        }

        // checkout input_indexs size equals input shape
        CHECK_EQ(input_indexs.size(), A->shape.size());
        return lang::CallExtern(reduce_type, {A(input_indexs)});
      },
      UniqName(output_name + "_tmp"));

  // compute output shape.
  std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + axes.front());
  int tailf = keep_dim ? (int(A->shape.size()) - axes.front()) : (int(A->shape.size()) - axes.back() - 1);
  for (int idx = 0; idx < tailf; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + axes.front());
        tmp_indexs.push_back(Expr(0));
        return tmp_out(tmp_indexs);
      },
      UniqName(output_name));
  return {out, tmp_out};
}

std::vector<ir::Tensor> BlockReduceSumInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, axes, keep_dim, "cinn_block_reduce_sum_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceProdInternal(const ir::Tensor& A,
                                                const std::vector<int>& axes,
                                                const bool keep_dim,
                                                const std::string& output_name) {
  return BlockReduceInternal(A, axes, keep_dim, "cinn_block_reduce_prod_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceMaxInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, axes, keep_dim, "cinn_block_reduce_max_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceMinInternal(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, axes, keep_dim, "cinn_block_reduce_min_internal", output_name);
}

/**
 * @brief compute the sum of array elements over the last dimension with block reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduce(const ir::Tensor& A,
                                    const std::vector<int>& axes,
                                    const int block_size,
                                    const bool keep_dim,
                                    const std::string& reduce_type,
                                    const std::string& output_name) {
  // compute reduce dimension size.
  Expr reduce_width(1);
  for (int idx = axes.front(); idx < A->shape.size(); ++idx) {
    reduce_width = reduce_width * A->shape[idx].as_int32();
  }

  // compute tmp output tensor shape
  std::vector<Expr> tmp_shape(A->shape.begin(), A->shape.begin() + axes.front());
  tmp_shape.push_back(Expr(block_size));
  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + axes.front());
        for (int idx = 0; idx < A->shape.size() - axes.front(); ++idx) {
          tmp_indexs.push_back(Expr(0));
        }
        // checkout input shape size equals tmp indexs size.
        CHECK_EQ(A->shape.size(), tmp_indexs.size());
        // compute offset.
        Expr offset = common::IndiceToAbsOffset(A->shape, tmp_indexs);
        // call block reduce sum
        return lang::CallExtern(reduce_type, {A, offset, reduce_width});
      },
      UniqName(output_name + "_tmp"));

  // compute output tensor shape.
  std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + axes.front());
  int tailf = keep_dim ? (int(A->shape.size()) - axes.front()) : (int(A->shape.size()) - axes.back() - 1);
  for (int idx = 0; idx < tailf; ++idx) {
    out_shape.push_back(Expr(1));
  }
  // if reduce on all dimension, the out_shape = {1}.
  if (out_shape.size() == 0) {
    out_shape.push_back(Expr(1));
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        // compute input index
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + axes.front());
        tmp_indexs.push_back(Expr(0));
        return tmp_out(tmp_indexs);
      },
      UniqName(output_name));

  return {out, tmp_out};
}

std::vector<ir::Tensor> BlockReduceSum(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, axes, block_size, keep_dim, "cinn_block_reduce_sum", output_name);
}

std::vector<ir::Tensor> BlockReduceProd(const ir::Tensor& A,
                                        const std::vector<int>& axes,
                                        const int block_size,
                                        const bool keep_dim,
                                        const std::string& output_name) {
  return BlockReduce(A, axes, block_size, keep_dim, "cinn_block_reduce_prod", output_name);
}

std::vector<ir::Tensor> BlockReduceMax(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, axes, block_size, keep_dim, "cinn_block_reduce_max", output_name);
}

std::vector<ir::Tensor> BlockReduceMin(const ir::Tensor& A,
                                       const std::vector<int>& axes,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, axes, block_size, keep_dim, "cinn_block_reduce_min", output_name);
}

int GetParallelThreads(const ir::Tensor& A, const std::vector<int>& axes) {
  int parallel_threads = 1;
  for (int idx = axes.back() + 1; idx < A->shape.size(); ++idx) {
    parallel_threads *= A->shape[idx].as_int32();
  }
  return parallel_threads;
}

ir::Tensor ReshapeInternal(const ir::Tensor& A, const std::vector<int>& axes, const std::string& output_name) {
  bool check_bound     = true;
  int last_stride      = A->shape[axes.back()].as_int32();
  int parallel_threads = GetParallelThreads(A, axes);
  int max_num_threads  = common::DefaultNVGPUTarget().max_num_threads();
  std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + axes.back());
  for (int idx = max_num_threads / parallel_threads; idx > ((max_num_threads / 2) / parallel_threads); --idx) {
    if (last_stride % idx == 0) {
      out_shape.emplace_back(last_stride / idx);
      out_shape.emplace_back(idx * parallel_threads);
      check_bound = false;
      break;
    }
  }

  if (check_bound) {
    int times = max_num_threads / parallel_threads;
    out_shape.emplace_back((last_stride + times - 1) / times);
    out_shape.emplace_back(times * parallel_threads);
  }

  std::vector<int> tail_strides(A->shape.size() - axes.back(), 1);
  for (int idx = tail_strides.size() - 2, index = A->shape.size() - 1; idx >= 0; --idx, --index) {
    tail_strides[idx] = tail_strides[idx + 1] * A->shape[index].as_int32();
  }

  int tail_elements = 1;
  for (int idx = axes.back(); idx < A->shape.size(); ++idx) {
    tail_elements *= A->shape[idx].as_int32();
  }
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        Expr index    = indexs[out_shape.size() - 2] * out_shape.back() + indexs.back();
        auto selected = ir::LT::Make(index, Expr(tail_elements));

        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + axes.back());
        // last and the second of last.
        for (auto tail_stride : tail_strides) {
          tmp_indexs.push_back(index / Expr(tail_stride));
          index = index % Expr(tail_stride);
        }

        CHECK_EQ(tmp_indexs.size(), A->shape.size()) << "Indexs size is not equal to Input shape!";
        if (check_bound) {
          return ir::Select::Make(selected, A(tmp_indexs), Expr(0.0f));
        } else {
          return A(tmp_indexs);
        }
      },
      UniqName(output_name));
  return out;
}

ir::Tensor BlockShuffleReduce(const ir::Tensor& A,
                              std::vector<Expr>& tail,
                              const std::string& reduce_type,
                              const std::string& output_name) {
  std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + A->shape.size() - 1);
  int stride = 1;
  for (auto& t : tail) {
    stride *= t.as_int32();
    out_shape.push_back(t);
  }

  CHECK(A->buffer.defined()) << "Buffer is not defined!";
  auto out = Compute(
      out_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        return lang::CallExtern(reduce_type, {A, A->shape.back(), Expr(stride)});
      },
      UniqName(output_name));
  return out;
}

#define BlockShuffleReduce(name, reduce_type, init_val)                                                         \
  std::vector<ir::Tensor> BlockShuffleReduce##name(                                                             \
      const ir::Tensor& A, const std::vector<int>& axes, const bool keep_dim, const std::string& output_name) { \
    if (GetParallelThreads(A, axes) > common::DefaultNVGPUTarget().max_num_threads() / 2) {                     \
      return {Reduce##name(A, axes, keep_dim, output_name)};                                                    \
    } else {                                                                                                    \
      auto reduce_reshape  = ReshapeInternal(A, axes, output_name + "_reshape");                                \
      auto reduce_internal = Reduce##name(reduce_reshape, axes, keep_dim, output_name + "_internal");           \
      reduce_internal->WithBuffer("shared");                                                                    \
      std::vector<Expr> tail(A->shape.begin() + axes.back() + 1, A->shape.end());                               \
      auto reduce_out = BlockShuffleReduce(reduce_internal, tail, reduce_type, output_name);                    \
      return {reduce_out, reduce_internal, reduce_reshape};                                                     \
    }                                                                                                           \
  }

BlockShuffleReduce(Sum, "block_shuffle_sum", 0.0f);
BlockShuffleReduce(Prod, "block_shuffle_prod", 1.0f);
BlockShuffleReduce(Max, "block_shuffle_max", -3.402823e+38f);
BlockShuffleReduce(Min, "block_shuffle_min", 3.402823e+38f);

bool WithoutLastDimInReduce(const std::vector<ir::Expr>& inshape, const std::vector<int>& axes) {
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int sum_last_axes = 1;
  for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    sum_last_axes *= inshape[idx].as_int32();
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
};

using ReduceFunc =
    std::function<ir::Tensor(const ir::Tensor&, const std::vector<int>&, const bool, const std::string&)>;
using BlockReduceFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor&, const std::vector<int>&, const bool, const std::string&)>;

std::vector<ir::Tensor> TwoStepBlockReduceInternal(const ir::Tensor& A,
                                                   const std::vector<int>& axes,
                                                   const bool keep_dim,
                                                   const std::string& output_name,
                                                   ReduceFunc reduce_func,
                                                   BlockReduceFunc block_reduce_func) {
  CHECK(!WithoutLastDimInReduce(A->shape, axes)) << "Can't find last axis in reduce!";

  int index = axes.size() - 2;
  for (; index >= 0; --index) {
    if (axes[index] != axes[index + 1] - 1) {
      break;
    }
  }
  std::vector<int> first_axes(axes.begin(), axes.begin() + index + 1);
  std::vector<int> second_axes(axes.begin() + index + 1, axes.end());

  int lane             = 1;
  auto max_num_threads = common::DefaultNVGPUTarget().max_num_threads();
  for (int idx = static_cast<int>(second_axes.size()) - 1; idx >= 0; --idx) {
    lane *= A->shape[second_axes[idx]].as_int32();
    if (lane >= max_num_threads / 2) {
      for (int idy = 0; idy < idx; ++idy) {
        first_axes.push_back(second_axes[idy]);
      }
      std::vector<int> tmp;
      for (int idy = idx; idy < second_axes.size(); ++idy) {
        tmp.push_back(second_axes[idy]);
      }
      second_axes = tmp;
      break;
    }
  }

  bool keep_dim_first      = keep_dim;
  bool keep_dim_second     = keep_dim;
  auto reduce_reshape_func = [&first_axes,
                              &keep_dim_first,
                              &second_axes,
                              &keep_dim_second,
                              A,
                              axes,
                              keep_dim,
                              output_name,
                              lane,
                              index,
                              max_num_threads]() {
    bool check_bound = true;
    std::vector<Expr> out_shape(A->shape.begin(), A->shape.begin() + second_axes.front());
    if (second_axes.size() == 1) {
      int times = 1;
      int tail  = max_num_threads;
      for (; tail >= max_num_threads / 2; --tail) {
        if (lane % tail == 0) {
          check_bound = false;
          break;
        }
      }
      if (!check_bound) {
        times = lane / tail;
        out_shape.emplace_back(times);
        out_shape.emplace_back(tail);
      } else {
        times = (lane + max_num_threads - 1) / max_num_threads;
        out_shape.emplace_back(times);
        out_shape.emplace_back(max_num_threads);
      }
    } else {
      int times = 1;
      int head  = A->shape[second_axes.front()].as_int32();
      int tail  = lane / head;
      // from (1024, 512) check one size as tail.
      for (int idx = (max_num_threads / tail); idx > (max_num_threads / 2 / tail); --idx) {
        if (head % idx == 0) {
          check_bound = false;
          times       = idx;
          tail *= idx;
          break;
        }
      }
      if (!check_bound) {
        out_shape.emplace_back(head / times);
        out_shape.emplace_back(tail);
      } else {
        times = max_num_threads / tail;
        out_shape.emplace_back((head + times - 1) / times);
        out_shape.emplace_back(tail * times);
      }
    }
    first_axes.push_back(out_shape.size() - 2);

    if (keep_dim) {
      second_axes = {static_cast<int>(out_shape.size()) - 1};
      if (out_shape.size() > A->shape.size()) {
        keep_dim_second = false;
      } else {
        keep_dim_second = true;
      }
      int tail_count = A->shape.size() - out_shape.size();
      for (int idx = 0; idx < tail_count; ++idx) {
        out_shape.push_back(Expr(1));
      }
    } else {
      second_axes = {static_cast<int>(out_shape.size()) - static_cast<int>(first_axes.size()) - 1};
    }

    std::vector<int> tail_strides(A->shape.size() - (out_shape.size() - 2), 1);
    for (int idx = static_cast<int>(tail_strides.size()) - 2, index = static_cast<int>(A->shape.size()) - 1; idx >= 0;
         --idx, --index) {
      tail_strides[idx] = tail_strides[idx + 1] * A->shape[index].as_int32();
    }

    auto out = Compute(
        out_shape,
        [=](const std::vector<Expr>& indexs) -> Expr {
          Expr index = indexs.back() + indexs[indexs.size() - 2] * out_shape.back();
          std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + out_shape.size() - 2);
          // last and the second of last.
          auto selected = ir::LT::Make(index, Expr(lane));
          for (auto tail_stride : tail_strides) {
            tmp_indexs.push_back(index / Expr(tail_stride));
            index = index % Expr(tail_stride);
          }

          CHECK_EQ(tmp_indexs.size(), A->shape.size()) << "Indexs size is not equal to Input shape!";
          if (check_bound) {
            return ir::Select::Make(selected, A(tmp_indexs), Expr(0.0f));
          } else {
            return A(tmp_indexs);
          }
        },
        UniqName(output_name + "_reshape"));
    return out;
  };
  std::vector<ir::Tensor> results;
  if (lane > max_num_threads) {
    results.push_back(reduce_reshape_func());
  } else {
    if (!keep_dim) {
      for (auto& axis : second_axes) {
        axis -= first_axes.size();
      }
    }
  }
  if (first_axes.size()) {
    results.push_back(
        reduce_func(results.size() ? results.back() : A, first_axes, keep_dim_first, output_name + "_internal"));
    results.back()->WithBuffer("local");
  }
  if (second_axes.size()) {
    auto res = block_reduce_func(results.size() ? results.back() : A, second_axes, keep_dim_second, output_name);
    res[1]->WithBuffer("local");
    results.push_back(res[1]);
    results.push_back(res[0]);
  }
  std::reverse(results.begin(), results.end());
  return results;
}

std::vector<ir::Tensor> TwoStepBlockReduceSum(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A, axes, keep_dim, output_name, ReduceSum, BlockReduceSumInternal);
}

std::vector<ir::Tensor> TwoStepBlockReduceProd(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return TwoStepBlockReduceInternal(A, axes, keep_dim, output_name, ReduceProd, BlockReduceProdInternal);
}

std::vector<ir::Tensor> TwoStepBlockReduceMax(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A, axes, keep_dim, output_name, ReduceMax, BlockReduceMaxInternal);
}

std::vector<ir::Tensor> TwoStepBlockReduceMin(const ir::Tensor& A,
                                              const std::vector<int>& axes,
                                              const bool keep_dim,
                                              const std::string& output_name) {
  return TwoStepBlockReduceInternal(A, axes, keep_dim, output_name, ReduceMin, BlockReduceMinInternal);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

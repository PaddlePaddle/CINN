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
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/ir/ir_constant.h"
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

Tensor ReduceSum(
    const Tensor& A, const std::vector<int>& axes, bool keep_dims, ir::Expr initial, const std::string& output_name) {
  if (!initial.defined()) {
    initial = common::make_const(A->type(), 0);
  }
  return Reduce(A, axes, lang::ReduceSum, keep_dims, initial, output_name);
}

Tensor ReduceProd(
    const Tensor& A, const std::vector<int>& axes, bool keep_dims, ir::Expr initial, const std::string& output_name) {
  if (!initial.defined()) {
    initial = common::make_const(A->type(), 1);
  }
  return Reduce(A, axes, lang::ReduceMul, keep_dims, initial, output_name);
}

Tensor ReduceMax(
    const Tensor& A, const std::vector<int>& axes, bool keep_dims, Expr initial, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceMax, keep_dims, Expr(), output_name);
}

Tensor ReduceMin(
    const Tensor& A, const std::vector<int>& axes, bool keep_dims, Expr initial, const std::string& output_name) {
  return Reduce(A, axes, lang::ReduceMin, keep_dims, Expr(), output_name);
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

  // compute tmp output shape.
  std::vector<Expr> tmp_shape(A->shape.begin(), A->shape.begin() + shape_size_without_reduce_dim);
  tmp_shape.push_back(reduce_width);

  // compute the reduce dimension stride.
  std::vector<Expr> last_reduce_stride(last_reduce_dim_num, Expr(1));
  for (int idx = A->shape.size(), index = last_reduce_stride.size() - 2; index >= 0; --index) {
    last_reduce_stride[index] = last_reduce_stride[index + 1] * A->shape[--idx];
  }

  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        // comput index map from output to input.
        auto last_index = indexs.back();
        std::vector<Expr> input_indexs(indexs.begin(), indexs.begin() + indexs.size() - 1);
        for (int idx = 0; idx < last_reduce_dim_num; ++idx) {
          input_indexs.push_back(last_index / last_reduce_stride[idx]);
          last_index = last_index % last_reduce_stride[idx];
        }

        // checkout input_indexs size equals input shape
        CHECK_EQ(input_indexs.size(), A->shape.size());
        return lang::CallExtern(reduce_type, {A(input_indexs)});
      },
      UniqName(output_name + "_tmp"));

  // compute output shape.
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

std::vector<ir::Tensor> BlockReduceSumInternal(const ir::Tensor& A,
                                               const int last_reduce_dim_num,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, last_reduce_dim_num, keep_dim, "cinn_block_reduce_sum_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceProdInternal(const ir::Tensor& A,
                                                const int last_reduce_dim_num,
                                                const bool keep_dim,
                                                const std::string& output_name) {
  return BlockReduceInternal(A, last_reduce_dim_num, keep_dim, "cinn_block_reduce_prod_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceMaxInternal(const ir::Tensor& A,
                                               const int last_reduce_dim_num,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, last_reduce_dim_num, keep_dim, "cinn_block_reduce_max_internal", output_name);
}

std::vector<ir::Tensor> BlockReduceMinInternal(const ir::Tensor& A,
                                               const int last_reduce_dim_num,
                                               const bool keep_dim,
                                               const std::string& output_name) {
  return BlockReduceInternal(A, last_reduce_dim_num, keep_dim, "cinn_block_reduce_min_internal", output_name);
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
                                    const int last_reduce_dim_num,
                                    const int block_size,
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

  // compute tmp output tensor shape
  std::vector<Expr> tmp_shape(A->shape.begin(), A->shape.begin() + shape_size_without_reduce_dim);
  tmp_shape.push_back(Expr(block_size));
  auto tmp_out = Compute(
      tmp_shape,
      [=](const std::vector<Expr>& indexs) -> Expr {
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + shape_size_without_reduce_dim);
        for (int idx = 0; idx < last_reduce_dim_num; ++idx) {
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
        // compute input index
        std::vector<Expr> tmp_indexs(indexs.begin(), indexs.begin() + shape_size_without_reduce_dim);
        tmp_indexs.push_back(Expr(0));
        return tmp_out(tmp_indexs);
      },
      UniqName(output_name));

  return {out, tmp_out};
}

std::vector<ir::Tensor> BlockReduceSum(const ir::Tensor& A,
                                       const int last_reduce_dim_num,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, last_reduce_dim_num, block_size, keep_dim, "cinn_block_reduce_sum", output_name);
}

std::vector<ir::Tensor> BlockReduceProd(const ir::Tensor& A,
                                        const int last_reduce_dim_num,
                                        const int block_size,
                                        const bool keep_dim,
                                        const std::string& output_name) {
  return BlockReduce(A, last_reduce_dim_num, block_size, keep_dim, "cinn_block_reduce_prod", output_name);
}

std::vector<ir::Tensor> BlockReduceMax(const ir::Tensor& A,
                                       const int last_reduce_dim_num,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, last_reduce_dim_num, block_size, keep_dim, "cinn_block_reduce_max", output_name);
}

std::vector<ir::Tensor> BlockReduceMin(const ir::Tensor& A,
                                       const int last_reduce_dim_num,
                                       const int block_size,
                                       const bool keep_dim,
                                       const std::string& output_name) {
  return BlockReduce(A, last_reduce_dim_num, block_size, keep_dim, "cinn_block_reduce_min", output_name);
}

std::vector<ir::Tensor> ReduceForSmallerDim(const ir::Tensor& A,
                                            const std::vector<int>& axes,
                                            const std::function<Expr(Expr, Expr)>& func,
                                            bool keep_dim,
                                            Expr initial,
                                            const std::string& output_name = "T_Reduce_out") {
  CHECK_NE(A->shape.size(), axes.size()) << "ReduceForLargerAxisDim not support reduce all ! Please check.";
  bool is_succesive = true;
  for (int i = axes.size() - 2; i >= 0; --i) {
    CHECK_EQ(axes[i] + 1, axes[i + 1]) << "ReduceForLargerAxisDim only supporting succesive axis, but here axis=["
                                       << cinn::utils::Join(axes, ",") << "] ! Please check.";
  }

  // for shape=[a, b, c, d, e, f], axis=[2, 3], shape[axis]=[c, d]
  // --> tmp_out=[a, b, c*d, 1, e, f]
  // --> out=[a, b, 1, 1, e, f] if keep_dim else [a, b, e, f]

  auto input_shape = ToShapeType(A->shape);

  // compute tensor size of each range
  auto shape_product = [&](const int begin, const int end) {
    int prod = 1;
    for (int i = begin; i < end; ++i) {
      prod *= input_shape[i];
    }
    return prod;
  };

  int axis_dim = shape_product(axes.front(), axes.back() + 1);
  int last_dim = shape_product(axes.back() + 1, A->shape.size());

  CHECK_LE(last_dim, 128) << "The subsequent dim of reduce should less than 128 to ensure concurrency, but here "
                          << last_dim << " > 128 ! Please check.";

  // When `e*f < 1024`, we can compute a reduce result in a block,
  // in other words, a block can compute multi reduce result.
  // For example, we can reshape the input into three parts,
  // [batch, row, col] where batch=a*b, row=c*d, col=e*f,
  // that means, we can reduce `row*col` number into row in a block with a loop,
  // where a loop can compute `row_in_block=1024/col` row,
  // each block loop `row_per_thread=row/row_in_block`.

  // we can compute it by two step:
  // loop1 :                 col_1 | col_2 | ... | col_[row_in_block]
  // ...                        +      +     ...         +
  // loop_[row_per_thread] : col_1 | col_2 | ... | col_[row_in_block]
  //   |
  //   V
  // tmp_out               : sum_1 | sum_2 | ... | sum_[row_in_block]
  //   |
  //   V
  // out = sum_1 + sum_2 + ... + sum_[row_in_block]

  int block_size = 1024;
  int row_in_block, row_per_thread;
  do {
    // compute the max row number in a block without loop
    row_in_block = block_size / last_dim;
    // compute the row number of each thread should compute
    row_per_thread = (axis_dim + row_in_block - 1) / row_in_block;
    block_size /= 2;
  } while (block_size > 0 && row_per_thread == 1);

  CHECK(block_size > 0) << "The size of reduce tensor should greater than 2 ! Please check.";

  VLOG(4) << "A: [" << cinn::utils::Join(input_shape, ",") << "] axes: (" << cinn::utils::Join(axes, ",")
          << ") row_in_block: " << row_in_block << " row_per_thread: " << row_per_thread;

  // compute output shape
  std::vector<ir::Expr> tmp_shape, out_shape;
  {
    auto compute_out_shape = [&](const int bein, const int end) {
      for (int i = bein; i < end; ++i) {
        tmp_shape.emplace_back(A->shape[i]);
        out_shape.emplace_back(A->shape[i]);
      }
    };

    compute_out_shape(0, axes.front());
    for (int i = 0; i < axes.size(); ++i) {
      tmp_shape.emplace_back(1);  // useless shape, only convenient for index compute
      if (keep_dim) {
        out_shape.emplace_back(1);
      }
    }
    tmp_shape[axes.front()] = Expr(row_in_block);
    compute_out_shape(axes.back() + 1, A->shape.size());

    if (out_shape.empty()) {
      // to avoid reduce all
      out_shape.emplace_back(1);
    }
  }

  // change output index to input index
  std::vector<int> prod_shape(axes.size(), 1);
  for (int i = axes.size() - 2; i >= 0; --i) {
    prod_shape[i] = prod_shape[i + 1] * input_shape[axes[i + 1]];
  }

  // tmp_index=[a, b, c*d, 1, e, f] --> input=[a, b, c, d, e, f]
  // change index of [c*d] to [c, d]
  auto outidx2inidx = [&](const std::vector<Expr>& output_index, Expr index) {
    auto input_index = output_index;
    for (int i = 0; i < axes.size(); ++i) {
      input_index[axes[i]] = index / Expr(prod_shape[i]);
      index                = index - input_index[axes[i]] * Expr(prod_shape[i]);
    }
    return input_index;
  };

  auto tmp_out = lang::Compute(
      tmp_shape,
      [&](const std::vector<Expr>& indices) -> Expr {
        Expr sum(initial);
        for (int i = 0; i < row_per_thread; ++i) {
          auto index       = Expr(row_in_block * i) + indices[axes.front()];
          auto input_index = outidx2inidx(indices, index);
          if (i < row_per_thread - 1) {
            sum = func(sum, A(input_index));
          } else {
            sum = func(sum, ir::Select::Make(index < Expr(axis_dim), A(input_index), initial));
          }
        }
        return sum;
      },
      cinn::UniqName(output_name + "_tmp"));

  auto out = lang::Compute(
      out_shape,
      [&](const std::vector<Expr>& indices) -> Expr {
        Expr sum(initial);
        auto input_idx = indices;
        if (!keep_dim) {
          input_idx.insert(input_idx.begin() + axes.front(), axes.size(), Expr(0));
        }
        for (int i = 0; i < row_in_block; ++i) {
          input_idx[axes.front()] = Expr(i);
          sum                     = func(sum, tmp_out(input_idx));
        }
        return sum;
      },
      cinn::UniqName(output_name));

  return {out, tmp_out};
}

std::vector<ir::Tensor> ReduceSumForSmallerDim(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               bool keep_dim,
                                               const std::string& output_name) {
  return ReduceForSmallerDim(A, axes, ir::Add::Make, keep_dim, ir::Zero(A->type()), output_name);
}

std::vector<ir::Tensor> ReduceProdForSmallerDim(const ir::Tensor& A,
                                                const std::vector<int>& axes,
                                                bool keep_dim,
                                                const std::string& output_name) {
  return ReduceForSmallerDim(A, axes, ir::Mul::Make, keep_dim, ir::One(A->type()), output_name);
}

std::vector<ir::Tensor> ReduceMaxForSmallerDim(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               bool keep_dim,
                                               const std::string& output_name) {
  return ReduceForSmallerDim(A, axes, ir::Max::Make, keep_dim, ir::Minimum(A->type()), output_name);
}

std::vector<ir::Tensor> ReduceMinForSmallerDim(const ir::Tensor& A,
                                               const std::vector<int>& axes,
                                               bool keep_dim,
                                               const std::string& output_name) {
  return ReduceForSmallerDim(A, axes, ir::Min::Make, keep_dim, ir::Maximum(A->type()), output_name);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

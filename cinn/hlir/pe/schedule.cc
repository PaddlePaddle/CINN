#include "cinn/hlir/pe/schedule.h"

#include <isl/cpp.h>

#include <functional>
#include <numeric>
#include <utility>

#include "cinn/optim/ir_simplify.h"
#include "cinn/poly/isl_utils.h"
namespace cinn {
namespace hlir {
namespace pe {

int GetInnerSplitter(int origin, int other_axis) {
  int two_exp = 1;
  while (origin % two_exp == 0) {
    two_exp *= 2;
  }
  two_exp = two_exp / 2;
  int a   = SplitEven(two_exp);
  int b   = two_exp / a;
  while (a * other_axis >= 1024 || b * other_axis >= 1024) {
    two_exp = two_exp / 2;
    a       = SplitEven(two_exp);
    b       = two_exp / a;
  }
  if (origin == two_exp) {
    return 2;
  }
  return origin / two_exp;
}

int SplitEven(int origin) {
  int res = 1;
  while (origin % res == 0 && res * res < origin) {
    res *= 2;
  }
  res = res / 2;
  return res;
}

int GetBasicFactor(const Type &type, const common::Target &target) {
  int target_native_vector_bits = target.get_target_bits() * 8;
  int type_bits                 = type.bits();
  return target_native_vector_bits / type_bits;
}

int GetBetterSplitFactor(int shape, int split_factor) {
  int better_factor = split_factor;
  while (better_factor > shape) {
    better_factor /= 2;
  }
  if (better_factor < shape && better_factor != split_factor) return better_factor * 2;
  return better_factor;
}

void ScheduleInjectiveCPU(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target) {
  int dims = stage->n_out_dims();
  if (dims > 1) {
    CHECK_EQ(stage->n_out_dims(), stage->n_in_dims()) << "The dims of op are not equal";
    CHECK_EQ(stage->n_out_dims(), output_shape.size())
        << "The origin stage out dims should be same with output_shape sizes";
    poly::Iterator fused          = stage->axis(dims - 1);
    int target_native_vector_bits = target.get_target_bits() * 8;
    int type_bits                 = stage->tensor()->type().bits();
    int prod_size                 = output_shape.back();
    // fuse conservatively for the complex index from poly and may not benefit a lot compared with llvm optimization,
    // only fuse the last two dims when the last dimension is too small and can split and vectorize Todo: try reorder
    if (output_shape.back() * type_bits < target_native_vector_bits) {
      int last_two_dim_bits = output_shape[dims - 2] * output_shape[dims - 1] * type_bits;
      if (last_two_dim_bits % target_native_vector_bits == 0) {
        fused = stage->Fuse(dims - 2, dims - 1);
        prod_size *= output_shape[dims - 2];
      }
    }
    int split_factor = target_native_vector_bits / type_bits;
    if (prod_size <= split_factor) {
      split_factor = GetBetterSplitFactor(prod_size, split_factor);
      if (split_factor >= 8) {
        stage->Vectorize(fused, split_factor);
      }
    } else {
      auto [j_outer, j_inner] = stage->Split(fused, split_factor);
      stage->Vectorize(j_inner, split_factor);
    }
  }
  if (stage->n_out_dims() > 1) {
    stage->Parallel(0);
  }
  return;
}

void CudaScheduleMul(poly::StageMap stages,
                     ir::Tensor output,
                     const std::vector<int> &output_shape,
                     const common::Target &target) {
  stages[output]->Split(1, 2);
  stages[output]->Bind(0, "blockIdx.x");
  stages[output]->Bind(1, "threadIdx.x");

  return;
}

int GetArrayPackingFactor(int shape, const Type &type, const common::Target &target) {
  int split_base   = GetBasicFactor(type, target);
  int split_factor = 1;
  // temporily use shape-1 instead of shape for isl wrong for1 elimination
  int i = split_base * split_base < shape ? split_base * split_base : shape;
  for (; i > 1; i--) {
    if (shape % i == 0) {
      split_factor = i;
      break;
    }
  }
  return split_factor;
}

void MatmulScheduleCPU(poly::StageMap stages,
                       const ir::Tensor &output,
                       const ir::Tensor &packedB,
                       const common::Target &target) {
  CHECK_EQ(output->type(), packedB->type());
  int basic_split_factor = GetBasicFactor(packedB->type(), target);
  // packedB
  int packedB_dims         = stages[packedB]->axis_names().size();
  int packed_last_dim      = packedB->shape[packedB_dims - 1].as_int32();
  int packedB_split_factor = GetBetterSplitFactor(packed_last_dim, basic_split_factor);
  // tempory solution for indivisible case
  if (packedB_split_factor >= 8 && packed_last_dim % packedB_split_factor == 0) {
    stages[packedB]->Vectorize(packedB_dims - 1, packedB_split_factor);
  }
  // output
  int output_size = output->shape.size();
  // M, N
  int M             = output->shape[output_size - 2].as_int32();
  int N             = output->shape[output_size - 1].as_int32();
  int bm            = GetArrayPackingFactor(M, output->type(), target);
  int bn            = GetArrayPackingFactor(N, output->type(), target);
  int out_axis_dims = stages[output]->axis_names().size();
  CHECK_GE(out_axis_dims, 3U) << "output tensor's size should be at least 3";
  poly::Iterator i_axis = stages[output]->axis(out_axis_dims - 3);
  poly::Iterator j_axis = stages[output]->axis(out_axis_dims - 2);
  poly::Iterator i_outer, i_inner, j_outer, j_inner;
  std::vector<poly::Iterator> i_axes, j_axes, k_axes;
  std::vector<poly::Iterator> all_axes;
  std::vector<poly::Iterator> all_axes_outer;
  std::vector<poly::Iterator> all_axes_inner;
  bool is_m_splited = false;
  bool is_n_splited = false;
  // tempory solution for isl for1 wrong elimination
  if (bm >= 4 && M != bm) {
    auto axes = stages[output]->Split(i_axis, bm);
    all_axes_outer.push_back(std::get<0>(axes));
    all_axes_inner.push_back(std::get<1>(axes));
    is_m_splited = true;
  } else {
    all_axes_outer.push_back(i_axis);
  }
  out_axis_dims = stages[output]->axis_names().size();
  // temp solution for isl for1 wrong elimination
  if (bn >= 4 && N != bn) {
    auto axes = stages[output]->Split(j_axis, bn);
    all_axes_outer.push_back(std::get<0>(axes));
    all_axes_inner.push_back(std::get<1>(axes));
    is_n_splited = true;
  } else {
    all_axes_outer.push_back(j_axis);
  }
  // K
  int K              = packedB->shape[packedB->shape.size() - 2].as_int32();
  int k_split_factor = GetBetterSplitFactor(K, basic_split_factor);
  out_axis_dims      = stages[output]->axis_names().size();
  auto k_axis        = stages[output]->axis(out_axis_dims - 1);
  bool is_k_splited  = false;
  if (k_split_factor >= 4) {
    auto axes = stages[output]->Split(k_axis, k_split_factor);
    k_axes.push_back(std::get<0>(axes));
    k_axes.push_back(std::get<1>(axes));
    all_axes_outer.push_back(std::get<0>(axes));
    all_axes_inner.push_back(std::get<1>(axes));
    is_k_splited = true;
  } else {
    all_axes_outer.push_back(k_axis);
  }
  std::vector<poly::Iterator> new_order;
  out_axis_dims = stages[output]->axis_names().size();
  if (output_size > 2) {
    // batch
    all_axes.push_back(stages[output]->axis(0));
  }
  for (int i = 0; i < all_axes_outer.size(); ++i) {
    all_axes.push_back(all_axes_outer[i]);
  }
  for (int i = 0; i < all_axes_inner.size(); ++i) {
    all_axes.push_back(all_axes_inner[i]);
  }
  // int axies
  CHECK_EQ(all_axes.size(), out_axis_dims);
  if (is_k_splited) {
    if (is_m_splited || is_n_splited) {
      // swap k_inner and j_inner/i_inner
      std::swap(all_axes[out_axis_dims - 1], all_axes[out_axis_dims - 2]);
    } else {
      // swap k_inner and j
      std::swap(all_axes[out_axis_dims - 1], all_axes[out_axis_dims - 3]);
    }
  } else {
    // swap k and j
    std::swap(all_axes[out_axis_dims - 1], all_axes[out_axis_dims - 2]);
  }
  stages[output]->Reorder(all_axes);
  // vectorize output's last dimemsion
  auto out_domain = stages[output]->transformed_domain();
  auto [min, max] = poly::isl_set_get_axis_range(out_domain.get(), out_axis_dims - 1);
  CHECK_EQ(min.get_num_si(), 0) << "axis range should begin from zero";
  int out_last_dim        = max.get_num_si() + 1;
  int output_split_factor = GetBetterSplitFactor(out_last_dim, basic_split_factor);
  // tempory solution for indivisible case
  if (output_split_factor >= 8 && packed_last_dim % output_split_factor == 0) {
    stages[output]->Vectorize(out_axis_dims - 1, output_split_factor);
  }
}

void MulScheduleCPU(poly::StageMap stages,
                    const ir::Tensor &output,
                    const ir::Tensor &reduce_first,
                    const common::Target &target) {
  int split_factor                     = GetBasicFactor(output->type(), target);
  auto out_reduce_axis                 = output->reduce_axis;
  std::vector<Expr> reduce_first_shape = reduce_first->shape;
  std::vector<Expr> output_shape       = output->shape;
  CHECK_EQ(reduce_first_shape.size(), 3U);
  CHECK_EQ(output_shape.size(), 2U);

  // reduce_first init
  auto reduce_first_init    = reduce_first->GetInitTensor(stages, target);
  int reduce_first_init_dim = stages[reduce_first_init]->axis_names().size();
  stages[reduce_first_init]->ComputeAt2(stages[reduce_first], reduce_first_init_dim - 2);
  // output init
  auto out_init    = output->GetInitTensor(stages, target);
  int out_init_dim = stages[out_init]->axis_names().size();
  stages[out_init]->ComputeAt2(stages[output], out_init_dim - 1);
  // reduce_first
  int reduce_first_dim = stages[reduce_first]->axis_names().size();
  stages[reduce_first]->Reorder({reduce_first_dim - 1, reduce_first_dim - 2});
  int reduce_first_last_shape = reduce_first_shape.back().as_int32();
  // output
  int out_dims = stages[output]->n_out_dims();
  if (reduce_first_last_shape > 1) {
    stages[output]->Unroll(out_dims - 1);
  }
}

void CudaScheduleConv(poly::StageMap stages,
                      ir::Tensor &input_pad,
                      ir::Tensor &kernel_dilation,
                      ir::Tensor &output,
                      const common::Target &target) {
  int n = output->shape[0].as_int32();
  int c = output->shape[1].as_int32();
  optim::Simplify(&(output->shape[2]));
  int h = output->shape[2].as_int32();
  optim::Simplify(&(output->shape[3]));
  int w  = output->shape[3].as_int32();
  int rc = kernel_dilation->shape[1].as_int32();
  int ry = kernel_dilation->shape[2].as_int32();
  int rx = kernel_dilation->shape[3].as_int32();

  int f_inner  = GetInnerSplitter(c, h);
  int block_z  = SplitEven(c / f_inner);
  int thread_z = c / f_inner / block_z;

  int rc_factor = SplitEven(rc);

  auto OL = stages[output]->CacheWrite2("local", stages, output);

  auto tx        = stages[output]->axis(3);
  auto by        = stages[output]->axis(2);
  auto [tem, fi] = stages[output]->Split(1, f_inner);
  auto [bz, tz]  = stages[output]->Split(1, thread_z);
  stages[output]->Reorder({bz, by, tz, tx, fi});
  stages[output]->Bind(1, "blockIdx.z");
  stages[output]->Bind(2, "blockIdx.y");
  stages[output]->Bind(3, "threadIdx.z");
  stages[output]->Bind(4, "threadIdx.x");
  stages[OL]->ComputeAt3(stages[output], 4);
  auto on  = stages[OL]->axis(0);
  auto obz = stages[OL]->axis(1);
  auto oby = stages[OL]->axis(2);
  auto otz = stages[OL]->axis(3);
  auto otx = stages[OL]->axis(4);
  auto ofi = stages[OL]->axis(5);
  auto orc = stages[OL]->axis(6);
  auto ory = stages[OL]->axis(7);
  auto orx = stages[OL]->axis(8);
  stages[OL]->Reorder({orc, ory, orx, on, obz, oby, otz, otx, ofi});
  if (rc_factor > 1) {
    stages[OL]->Split(0, rc_factor);
    stages[OL]->Bind(5, "blockIdx.z");
    stages[OL]->Bind(6, "blockIdx.y");
    stages[OL]->Bind(7, "threadIdx.z");
    stages[OL]->Bind(8, "threadIdx.x");
  } else {
    stages[OL]->Bind(4, "blockIdx.z");
    stages[OL]->Bind(5, "blockIdx.y");
    stages[OL]->Bind(6, "threadIdx.z");
    stages[OL]->Bind(7, "threadIdx.x");
  }

  return;
}

void CudaScheduleInjective(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target) {
  CHECK_EQ(stage->n_out_dims(), stage->n_in_dims()) << "The dims of op are not equal";
  int dims = stage->n_out_dims();
  for (int i = 1; i < dims; i++) {
    stage->Fuse(0, 1);
  }
  int num_thread        = target.max_num_threads();
  int num_block         = 256;
  int vector_width      = 1;
  int prod_size         = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  bool need_block_split = prod_size > num_thread * num_block * vector_width ? true : false;
  if (need_block_split) {
    auto [X_outer, X_inner]  = stage->Split(0, num_thread * num_block);
    auto [Block_x, Thread_x] = stage->Split(X_inner, num_thread);
    stage->Reorder({Block_x, Thread_x, X_outer});
    stage->Bind(0, "blockIdx.x");
    stage->Bind(1, "threadIdx.x");
  } else {
    if (prod_size > num_thread) {
      stage->Split(0, num_thread);
      stage->Bind(0, "blockIdx.x");
      stage->Bind(1, "threadIdx.x");
    } else {
      stage->Bind(0, "threadIdx.x");
    }
  }
  return;
}

void CudaSplitSchedule(poly::Stage *stage, const std::vector<int> &output_shape) {
  if (output_shape.size() > 1 && output_shape[1] >= 512) {
    int temp_split = 1;
    int temp_num   = output_shape[1];
    while (temp_num >= 512) {
      temp_split = temp_split * 2;
      temp_num   = temp_num / 2;
    }
    stage->Split(1, temp_split);
  }
  return;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

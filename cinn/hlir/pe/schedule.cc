#include "cinn/hlir/pe/schedule.h"

#include <functional>
#include <numeric>

namespace cinn {
namespace hlir {
namespace pe {

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
                      ir::Tensor input_pad,
                      ir::Tensor kernel_dilation,
                      ir::Tensor output,
                      const common::Target &target) {
  int num_thread = target.max_num_threads();
  stages[output]->Fuse(0, 1);
  auto [Block_x, Thread_x] = stages[output]->Split(0, num_thread);
  stages[output]->Bind(0, "blockIdx.x");
  stages[output]->Bind(1, "threadIdx.x");

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

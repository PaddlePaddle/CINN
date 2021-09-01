#include "cinn/hlir/pe/schedule.h"

#include <isl/cpp.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "cinn/common/cas.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/poly/isl_utils.h"

namespace cinn {
namespace hlir {
namespace pe {

ScheduleParam::ScheduleParam() {}

ScheduleParam::~ScheduleParam() {}

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
      auto[j_outer, j_inner] = stage->Split(fused, split_factor);
      stage->Vectorize(j_inner, split_factor);
    }
  }
  if (stage->n_out_dims() > 1) {
    stage->Parallel(0);
  }
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
  auto[min, max] = poly::isl_set_get_axis_range(out_domain.get(), out_axis_dims - 1);
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

void PoolScheduleGPU(poly::StageMap stages, ir::Tensor &output, const common::Target &target) {
  CHECK_GE(stages[output]->axis_names().size(), 4);
  stages[output]->Fuse({0, 1, 2, 3});
  stages[output]->Split(0, 1024);
  stages[output]->Bind(0, "blockIdx.x");
  stages[output]->Bind(1, "threadIdx.x");
}

void GetConv2dFactors(std::unordered_map<std::string, int> *factors,
                      int oc,
                      int ic,
                      int ow,
                      const Type &type,
                      const common::Target &target) {
  int bn_base = GetBasicFactor(type, target);
  int oc_bn   = 1;
  for (int i = bn_base; i > 1; i--) {
    if (oc < 1) break;
    if (oc % i == 0) {
      oc_bn = i;
      break;
    }
  }
  int ic_bn = 1;
  for (int i = oc_bn; i > 1; i--) {
    if (ic < 1) break;
    if (ic % i == 0) {
      ic_bn = i;
      break;
    }
  }
  int ow_bn = 1;
  for (int i = bn_base; i > 1; i--) {
    if (ow < 1) break;
    if (ow % i == 0) {
      ow_bn = i;
      break;
    }
  }

  (*factors)["oc_bn"] = oc_bn;
  (*factors)["ic_bn"] = ic_bn;
  (*factors)["ow_bn"] = ow_bn;
}

void GetConv2d1x1Factors(std::unordered_map<std::string, int> *factors,
                         int oc,
                         int ic,
                         int oh,
                         int ow,
                         const Type &type,
                         const common::Target &target) {
  int bn_base = GetBasicFactor(type, target);
  int oc_bn   = 1;
  for (int i = bn_base; i > 1; i--) {
    if (oc < 1) break;
    if (oc % i == 0) {
      oc_bn = i;
      break;
    }
  }
  int ic_bn = 1;
  for (int i = oc_bn; i > 1; i--) {
    if (ic < 1) break;
    if (ic % i == 0) {
      ic_bn = i;
      break;
    }
  }
  (*factors)["oc_bn"] = oc_bn;
  (*factors)["ic_bn"] = ic_bn;
  int ow_bn           = 1;
  int oh_bn           = 1;
  int begin           = std::min(ow, bn_base);
  for (int i = begin; i >= 1; i--) {
    if (ow < 1) break;
    if (ow % i == 0) {
      ow_bn = i;
      for (int j = oh; j >= 1; j--) {
        if (oh % j == 0 && j * ow_bn <= 16) {
          oh_bn               = j;
          (*factors)["oh_bn"] = oh_bn;
          (*factors)["ow_bn"] = ow_bn;
          return;
        }
      }
    }
  }
}

void Conv2d_NCHWc_1X1_Schedule_CPU(poly::StageMap stages,
                                   const ir::Tensor &res,
                                   ir::Tensor &packed_out,
                                   const ir::Tensor &input_pad,
                                   const ir::Tensor &weights_dilation,
                                   const ir::Tensor &data,
                                   const common::Target &target) {
  CHECK(target.arch == Target::Arch::X86) << "Conv2d_NCHWc_Schedule_CPU schedule only used in x86";
  CHECK(packed_out.defined());
  CHECK(input_pad.defined());
  auto type = packed_out->type();
  std::unordered_map<std::string, int> conv2d_factors;
  CHECK_EQ(packed_out->shape.size(), 5U) << "packed_out's shape size should be 5";
  Expr h_out             = common::AutoSimplify(packed_out->shape[2]);
  Expr w_out             = common::AutoSimplify(packed_out->shape[3]);
  int oh                 = h_out.as_int32();
  int ow                 = w_out.as_int32();
  int basic_split_factor = GetBasicFactor(type, target);
  GetConv2d1x1Factors(&conv2d_factors, -1, -1, oh, ow, type, target);
  int oh_bn_size = conv2d_factors["oh_bn"];
  int ow_bn_size = conv2d_factors["ow_bn"];

  auto input_shape = input_pad->shape;
  CHECK_EQ(input_shape.size(), 5U) << "input shape size should be 5";
  Expr oc_bn     = common::AutoSimplify(packed_out->shape.back());
  Expr ic_bn     = common::AutoSimplify(input_shape.back());
  int oc_bn_size = oc_bn.as_int32();
  int ic_bn_size = ic_bn.as_int32();
  VLOG(4) << "ow_bn_size" << ow_bn_size;
  VLOG(4) << "oc_bn_size" << oc_bn_size;
  VLOG(4) << "ic_bn_size" << ic_bn_size;

  // data
  if (data.defined()) {
    CHECK_GE(stages[data]->n_out_dims(), 3U) << "data's out_dims should be more than 3";
    stages[data]->Fuse({0, 1, 2});
    stages[data]->ComputeInline();
  }
  // input_pad
  CHECK_GE(stages[input_pad]->n_out_dims(), 3U) << "input_pad's out_dims should be more than 3";
  stages[input_pad]->Fuse({0, 1, 2});

  // weights
  if (weights_dilation.defined()) {
    CHECK_GE(stages[weights_dilation]->n_out_dims(), 3U) << "weights_dilation's out_dims should be more than 3";
    // oc_outer, ic_outer, oh, ow, ic_inner, oc_inner -> oc_outer, oh, ic_outer, ow, ic_inner, oc_inner
    stages[weights_dilation]->Reorder({2, 1});
    stages[weights_dilation]->Fuse({0, 1});
  }

  // packed_out
  auto CC = stages[packed_out]->CacheWrite("global", stages, packed_out);
  VLOG(4) << "ow_bn_size" << ow_bn_size;
  // packed_out: [batch, oc_outer, oh, ow, oc_inner]
  // split oh, ow
  stages[packed_out]->Split(2, oh_bn_size);
  stages[packed_out]->Split(4, ow_bn_size);
  // [batch, oc_outer, oh_outer, oh_inner, ow_outer, ow_inner, oc_inner] ->
  // [batch_oc_outer_oh_outer_fused, oh_inner, ow_outer, ow_inner, oc_inner]
  stages[packed_out]->Fuse({0, 1, 2});
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();

  // CC: [batch, oh, ow, oc, ic, kh, kw] -> [batch_oc_outer_oh_outer_fused, oh_inner, ow, oc_inner, ic, kh, kw]
  stages[CC]->ComputeAt2(stages[packed_out], 0);
  // tempory solution because reorder may be wrong before ComputeAt
  // reorder: [batch_oc_outer_oh_outer_fused, oh_inner, ow_outer, ow_inner, oc_inner] ->
  // [batch_oc_outer_oh_outer_fused, ow_outer, oh_inner, ow_inner, oc_inner]
  stages[packed_out]->Reorder({2, 1});
  stages[packed_out]->Vectorize(stages[packed_out]->n_out_dims() - 1, packed_out->shape.back().as_int32());
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();

  // CC: [batch_oc_outer_oh_outer_fused, oh_inner, ow, oc_inner, ic, kh, kw]
  // split ow
  stages[CC]->Split(2, ow_bn_size);
  // reorder: [batch_oc_outer_oh_outer_fused, oh_inner, ow_outer, ow_inner, oc_inner, ic, kh, kw] ->
  // [batch_oc_outer_oh_outer_fused, oh_inner, ow_outer, ow_inner, oc_inner, ic, kh, kw]
  stages[CC]->Reorder({2, 1});

  // split ic
  // CC: [batch_oc_outer_oh_outer_fused, ow_outer, oh_inner, ow_inner, oc_inner, ic, kh, kw]
  stages[CC]->Split(5, ic_bn_size);
  // reorder: [batch_oc_outer_oh_outer_fused, ow_outer, oh_inner, ow_inner, oc_inner, ic_outer, ic_inner, kh, kw] ->
  // [batch_oc_outer_oh_outer_fused, ow_outer, ic_outer, ic_inner, oh_inner, ow_inner, oc_inner, kh, kw]
  auto oh_inner = stages[CC]->axis(2);
  auto ow_inner = stages[CC]->axis(3);
  auto oc_inner = stages[CC]->axis(4);
  auto ic_outer = stages[CC]->axis(5);
  auto ic_inner = stages[CC]->axis(6);
  stages[CC]->Reorder({ic_outer, ic_inner, oh_inner, ow_inner, oc_inner});
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  stages[CC]->Vectorize(stages[CC]->n_out_dims() - 3, CC->shape.back().as_int32());
  // unroll ow_inner, oh_inner
  VLOG(4) << stages[CC]->transformed_domain();
  stages[CC]->Unroll(stages[CC]->n_out_dims() - 1);
  stages[CC]->Unroll(stages[CC]->n_out_dims() - 2);
  // CC_init
  auto CC_init = CC->GetInitTensor(stages, target);
  stages[CC_init]->Vectorize(stages[CC_init]->n_out_dims() - 1, CC_init->shape.back().as_int32());

  // res
  // n, oc, oh, ow
  if (res.defined()) {
    stages[res]->Split(1, oc_bn_size);
    stages[res]->Split(3, oh_bn_size);
    stages[res]->Split(5, ow_bn_size);
    // reorder: [n, oc_outer, oc_inner, oh_outer, oh_inner, ow_outer, ow_inner] ->
    // [n, oc_outer, oh_outer, ow_outer, oh_inner, ow_inner, oc_inner]
    auto oc_inner1 = stages[res]->axis(2);
    auto oh_outer1 = stages[res]->axis(3);
    auto oh_inner1 = stages[res]->axis(4);
    auto ow_outer1 = stages[res]->axis(5);
    auto ow_inner1 = stages[res]->axis(6);
    stages[res]->Reorder({oh_outer1, ow_outer1, oh_inner1, ow_inner1, oc_inner1});
    stages[res]->Fuse({0, 1, 2});
    VLOG(4) << "stages[res]->transformed_domain()" << stages[res]->transformed_domain();
  }
}

void Conv2d_NCHWc_1X1_Schedule_CPU_Nofuse(poly::StageMap stages,
                                          const ir::Tensor &res,
                                          ir::Tensor &packed_out,
                                          const ir::Tensor &input_pad,
                                          const ir::Tensor &weights_dilation,
                                          const ir::Tensor &data,
                                          const common::Target &target) {
  CHECK(target.arch == Target::Arch::X86) << "Conv2d_NCHWc_Schedule_CPU schedule only used in x86";
  CHECK(packed_out.defined());
  CHECK(input_pad.defined());
  auto type = packed_out->type();
  std::unordered_map<std::string, int> conv2d_factors;
  CHECK_EQ(packed_out->shape.size(), 5U) << "packed_out's shape size should be 5";
  Expr h_out             = common::AutoSimplify(packed_out->shape[2]);
  Expr w_out             = common::AutoSimplify(packed_out->shape[3]);
  int oh                 = h_out.as_int32();
  int ow                 = w_out.as_int32();
  int basic_split_factor = GetBasicFactor(type, target);
  GetConv2d1x1Factors(&conv2d_factors, -1, -1, oh, ow, type, target);
  int oh_bn_size = conv2d_factors["oh_bn"];
  int ow_bn_size = conv2d_factors["ow_bn"];

  auto input_shape = input_pad->shape;
  int shape_size   = input_shape.size();
  CHECK_EQ(shape_size, 5U) << "input shape size should be 5";
  Expr oc_bn     = common::AutoSimplify(packed_out->shape.back());
  Expr ic_bn     = common::AutoSimplify(input_shape.back());
  int oc_bn_size = oc_bn.as_int32();
  int ic_bn_size = ic_bn.as_int32();
  VLOG(4) << "ow_bn_size" << ow_bn_size;
  VLOG(4) << "oc_bn_size" << oc_bn_size;
  VLOG(4) << "ic_bn_size" << ic_bn_size;

  // data
  if (data.defined()) {
    stages[data]->ComputeInline();
  }
  // weights
  if (weights_dilation.defined()) {
    CHECK_GE(stages[weights_dilation]->n_out_dims(), 3U) << "weights_dilation's out_dims should be more than 3";
    // Reorder: [oc_outer, ic_outer, oh, ow, ic_inner, oc_inner] ->
    // [oc_outer, oh, ic_outer, ow, ic_inner, oc_inner]
    stages[weights_dilation]->Reorder({2, 1});
  }

  // packed_out
  auto CC = stages[packed_out]->CacheWrite("global", stages, packed_out);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // packed_out: [batch, oc_outer, oh, ow, oc_inner]
  // split oh, ow
  stages[packed_out]->Split(2, oh_bn_size);
  stages[packed_out]->Split(4, ow_bn_size);

  // CC: [batch, oc_outer, oh, ow, oc_inner]
  // packed_out: [batch, oc_outer, oh_outer, oh_inner, ow_outer, ow_inner, oc_inner]
  stages[CC]->ComputeAt2(stages[packed_out], 2);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // tempory solution because reordering before computeAt may be wrong
  // reorder: [batch, oc_outer, oh_outer, oh_inner, ow_outer, ow_inner, oc_inner] ->
  // [batch, oc_outer, oh_outer, ow_outer, oh_inner, ow_inner, oc_inner]
  stages[packed_out]->Reorder({4, 3});
  stages[packed_out]->Vectorize(stages[packed_out]->n_out_dims() - 1, packed_out->shape.back().as_int32());

  // split oh, ow
  // CC: [batch, oc_outer, oh_outer, oh_inner, ow, oc_inner, ic, kh, kw]
  stages[CC]->Split(4, ow_bn_size);
  // CC: [batch, oc_outer, oh_outer, oh_inner, ow_outer, ow_inner, oc_inner, ic, kh, kw]
  // split ic
  stages[CC]->Split(7, ic_bn_size);

  // reorder: [batch, oc_outer, oh_outer, oh_inner, ow_outer, ow_inner, oc_inner, ic_outer, ic_inner, kh, kw] ->
  // [batch, oc_outer, oh_outer, ow_outer, ic_outer, ic_inner, oh_inner, ow_inner, oc_inner, kh, kw]
  auto oh_inner = stages[CC]->axis(3);
  auto ow_outer = stages[CC]->axis(4);
  auto ow_inner = stages[CC]->axis(5);
  auto oc_inner = stages[CC]->axis(6);
  auto ic_outer = stages[CC]->axis(7);
  auto ic_inner = stages[CC]->axis(8);
  stages[CC]->Reorder({ow_outer, ic_outer, ic_inner, oh_inner, ow_inner, oc_inner});
  stages[CC]->Vectorize(stages[CC]->n_out_dims() - 3, CC->shape.back().as_int32());
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // CC_init
  auto CC_init = CC->GetInitTensor(stages, target);
  stages[CC_init]->Vectorize(stages[CC_init]->n_out_dims() - 1, CC_init->shape.back().as_int32());

  // res
  // n, oc, oh, ow
  if (res.defined()) {
    stages[res]->Split(1, oc_bn_size);
    stages[res]->Split(3, oh_bn_size);
    stages[res]->Split(5, ow_bn_size);
    // reorder: [n, oc_outer, oc_inner, oh_outer, oh_inner, ow_outer, ow_inner] ->
    // [n, oc_outer, oh_outer, ow_outer, oh_inner, ow_inner, oc_inner]
    auto oc_inner1 = stages[res]->axis(2);
    auto oh_outer1 = stages[res]->axis(3);
    auto oh_inner1 = stages[res]->axis(4);
    auto ow_outer1 = stages[res]->axis(5);
    auto ow_inner1 = stages[res]->axis(6);
    stages[res]->Reorder({oh_outer1, ow_outer1, oh_inner1, ow_inner1, oc_inner1});
    VLOG(4) << "stages[res]->transformed_domain()" << stages[res]->transformed_domain();
  }
}

void Conv2d_NCHWc_Schedule_CPU_Nofuse(poly::StageMap stages,
                                      const ir::Tensor &res,
                                      ir::Tensor &packed_out,
                                      const ir::Tensor &input_pad,
                                      const ir::Tensor &weights_dilation,
                                      const ir::Tensor &data,
                                      const common::Target &target) {
  CHECK(target.arch == Target::Arch::X86) << "Conv2d_NCHWc_Schedule_CPU schedule only used in x86";
  CHECK(packed_out.defined());
  CHECK(input_pad.defined());
  auto type = packed_out->type();
  std::unordered_map<std::string, int> conv2d_factors;
  CHECK_EQ(packed_out->shape.size(), 5U) << "packed_out's shape size should be 5";
  Expr w_out             = common::AutoSimplify(packed_out->shape[3]);
  int ow                 = w_out.as_int32();
  int basic_split_factor = GetBasicFactor(type, target);
  GetConv2dFactors(&conv2d_factors, -1, -1, ow, type, target);
  int ow_bn_size = conv2d_factors["ow_bn"];

  auto input_shape = input_pad->shape;
  int shape_size   = input_shape.size();
  CHECK_EQ(shape_size, 5U) << "input shape size should be 5";
  Expr oc_bn     = common::AutoSimplify(packed_out->shape.back());
  Expr ic_bn     = common::AutoSimplify(input_shape.back());
  int oc_bn_size = oc_bn.as_int32();
  int ic_bn_size = ic_bn.as_int32();
  VLOG(4) << "ow_bn_size " << ow_bn_size;
  VLOG(4) << "oc_bn_size " << oc_bn_size;
  VLOG(4) << "ic_bn_size " << ic_bn_size;

  // data
  if (data.defined()) {
    stages[data]->ComputeInline();
  }
  // weights
  if (weights_dilation.defined()) {
    CHECK_GE(stages[weights_dilation]->n_out_dims(), 3U) << "weights_dilation's out_dims should be more than 3";
    // Reorder: [oc_outer, ic_outer, oh, ow, ic_inner, oc_inner] ->
    // [oc_outer, oh, ic_outer, ow, ic_inner, oc_inner]
    stages[weights_dilation]->Reorder({2, 1});
  }
  // packed_out
  auto CC = stages[packed_out]->CacheWrite("global", stages, packed_out);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // packed_out: [batch, oc_outer, oh, ow, oc_inner]
  // split ow
  stages[packed_out]->Split(3, ow_bn_size);
  stages[packed_out]->Vectorize(stages[packed_out]->n_out_dims() - 1, packed_out->shape.back().as_int32());

  // CC: [batch, oc_outer, oh, ow, oc_inner]
  // packed_out: [batch, oc_outer, oh, ow_outer, ow_inner, oc_inner]
  // not computeAt ow_outer but oh
  stages[CC]->ComputeAt2(stages[packed_out], 2);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // split ow
  stages[CC]->Split(3, ow_bn_size);
  // CC: [batch, oc_outer, oh, ow_outer, ow_inner, oc_inner, ic, kh, kw]
  // split ic
  stages[CC]->Split(6, ic_bn_size);
  // reorder: [batch, oc_outer, oh, ow_outer, ow_inner, oc_inner, ic_outer, ic_inner, kh, kw] ->
  // [batch, oc_outer, oh, ow_outer, ic_outer, kh, kw, ic_inner, ow_inner, oc_inner]
  auto ow_inner = stages[CC]->axis(4);
  auto oc_inner = stages[CC]->axis(5);
  auto ic_outer = stages[CC]->axis(6);
  auto ic_inner = stages[CC]->axis(7);
  auto kh       = stages[CC]->axis(8);
  auto kw       = stages[CC]->axis(9);
  stages[CC]->Reorder({ic_outer, kh, kw, ic_inner, ow_inner, oc_inner});
  stages[CC]->Vectorize(stages[CC]->n_out_dims() - 1, CC->shape.back().as_int32());
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // CC_init
  auto CC_init = CC->GetInitTensor(stages, target);
  stages[CC_init]->Vectorize(stages[CC_init]->n_out_dims() - 1, CC_init->shape.back().as_int32());

  // res
  // n, oc, oh, ow
  if (res.defined()) {
    stages[res]->Split(1, oc_bn_size);
    stages[res]->Split(4, ow_bn_size);
    // Reorder: [n, oc_outer, oc_inner, oh, ow_outer, ow_inner] ->
    // [n, oc_outer, oh, ow_outer, ow_inner, oc_inner]
    auto oc_inner1 = stages[res]->axis(2);
    auto oh1       = stages[res]->axis(3);
    auto ow_outer1 = stages[res]->axis(4);
    auto ow_inner1 = stages[res]->axis(5);
    stages[res]->Reorder({oh1, ow_outer1, ow_inner1, oc_inner1});
    VLOG(4) << "stages[res]->transformed_domain()" << stages[res]->transformed_domain();
  }
}

void Conv2d_NCHWc_Schedule_CPU(poly::StageMap stages,
                               const ir::Tensor &res,
                               ir::Tensor &packed_out,
                               const ir::Tensor &input_pad,
                               const ir::Tensor &weights_dilation,
                               const ir::Tensor &data,
                               const common::Target &target) {
  CHECK(target.arch == Target::Arch::X86) << "Conv2d_NCHWc_Schedule_CPU schedule only used in x86";
  CHECK(packed_out.defined());
  CHECK(input_pad.defined());
  auto type = packed_out->type();
  std::unordered_map<std::string, int> conv2d_factors;
  CHECK_EQ(packed_out->shape.size(), 5U) << "packed_out's shape size should be 5";
  Expr w_out             = common::AutoSimplify(packed_out->shape[3]);
  int ow                 = w_out.as_int32();
  int basic_split_factor = GetBasicFactor(type, target);
  GetConv2dFactors(&conv2d_factors, -1, -1, ow, type, target);
  int ow_bn_size = conv2d_factors["ow_bn"];

  auto input_shape = input_pad->shape;
  int shape_size   = input_shape.size();
  CHECK_EQ(shape_size, 5U) << "input shape size should be 5";
  Expr oc_bn     = common::AutoSimplify(packed_out->shape.back());
  Expr ic_bn     = common::AutoSimplify(input_shape.back());
  int oc_bn_size = oc_bn.as_int32();
  int ic_bn_size = ic_bn.as_int32();
  VLOG(4) << "ow_bn_size" << ow_bn_size;
  VLOG(4) << "oc_bn_size" << oc_bn_size;
  VLOG(4) << "ic_bn_size" << ic_bn_size;

  // data
  if (data.defined()) {
    CHECK_GE(stages[data]->n_out_dims(), 3U) << "data's out_dims should be more than 3";
    stages[data]->Fuse({0, 1, 2});
    stages[data]->ComputeInline();
  }
  // input_pad
  CHECK_GE(stages[input_pad]->n_out_dims(), 3U) << "input_pad's out_dims should be more than 3";
  stages[input_pad]->Fuse({0, 1, 2});
  // weights
  if (weights_dilation.defined()) {
    CHECK_GE(stages[weights_dilation]->n_out_dims(), 3U) << "weights_dilation's out_dims should be more than 3";
    // oc_outer, ic_outer, oh, ow, ic_inner, oc_inner -> oc_outer, oh, ic_outer, ow, ic_inner, oc_inner
    stages[weights_dilation]->Reorder({2, 1});
    stages[weights_dilation]->Fuse({0, 1});
  }
  // packed_out
  auto CC = stages[packed_out]->CacheWrite("global", stages, packed_out);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // packed_out: [batch, oc_outer, oh, ow, oc_inner]
  // split ow
  stages[packed_out]->Split(3, ow_bn_size);
  stages[packed_out]->Fuse({0, 1, 2});
  stages[packed_out]->Vectorize(stages[packed_out]->n_out_dims() - 1, packed_out->shape.back().as_int32());

  // CC
  stages[CC]->ComputeAt2(stages[packed_out], 1);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // CC: [batch_oc_outer_oh_fused, ow_outer, ow_inner, oc_inner, ic, kh, kw]
  // for fused_axes' copy transform, not split ow again
  // split ic
  stages[CC]->Split(4, ic_bn_size);
  // reorder: [batch_oc_outer_oh_fused, ow_outer, ow_inner, oc_inner, ic_outer, ic_inner, kh, kw] ->
  // [batch_oc_outer_oh_fused, ow_outer, ic_outer, kh, kw, ic_inner, ow_inner, oc_inner]
  auto ow_inner = stages[CC]->axis(2);
  auto oc_inner = stages[CC]->axis(3);
  auto ic_outer = stages[CC]->axis(4);
  auto ic_inner = stages[CC]->axis(5);
  auto kh       = stages[CC]->axis(6);
  auto kw       = stages[CC]->axis(7);
  stages[CC]->Reorder({ic_outer, kh, kw, ic_inner, ow_inner, oc_inner});
  stages[CC]->Vectorize(stages[CC]->n_out_dims() - 1, CC->shape.back().as_int32());
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // CC_init
  auto CC_init = CC->GetInitTensor(stages, target);
  stages[CC_init]->Vectorize(stages[CC_init]->n_out_dims() - 1, CC_init->shape.back().as_int32());

  // res
  // n, oc, oh, ow
  if (res.defined()) {
    stages[res]->Split(1, oc_bn_size);
    stages[res]->Split(4, ow_bn_size);
    // Reorder: [n, oc_outer, oc_inner, oh, ow_outer, ow_inner] ->
    // [n, oc_outer, oh, ow_outer, ow_inner, oc_inner]
    auto oc_inner1 = stages[res]->axis(2);
    auto oh1       = stages[res]->axis(3);
    auto ow_outer1 = stages[res]->axis(4);
    auto ow_inner1 = stages[res]->axis(5);
    stages[res]->Reorder({oh1, ow_outer1, ow_inner1, oc_inner1});
    stages[res]->Fuse({0, 1, 2});
  }
}

void Depthwise_Conv2d_NCHWc_Schedule_CPU_Nofuse(poly::StageMap stages,
                                                const ir::Tensor &res,
                                                ir::Tensor &packed_out,
                                                const ir::Tensor &input_pad,
                                                const ir::Tensor &weights_dilation,
                                                const ir::Tensor &data,
                                                const common::Target &target) {
  CHECK(target.arch == Target::Arch::X86) << "Conv2d_NCHWc_Schedule_CPU schedule only used in x86";
  CHECK(packed_out.defined());
  CHECK(input_pad.defined());
  auto type = packed_out->type();
  std::unordered_map<std::string, int> conv2d_factors;
  CHECK_EQ(packed_out->shape.size(), 5U) << "packed_out's shape size should be 5";
  Expr w_out             = common::AutoSimplify(packed_out->shape[3]);
  int ow                 = w_out.as_int32();
  int basic_split_factor = GetBasicFactor(type, target);
  GetConv2dFactors(&conv2d_factors, -1, -1, ow, type, target);
  int ow_bn_size = conv2d_factors["ow_bn"];

  auto input_shape = input_pad->shape;
  int shape_size   = input_shape.size();
  CHECK_EQ(shape_size, 5U) << "input shape size should be 5";
  Expr oc_bn     = common::AutoSimplify(packed_out->shape.back());
  Expr ic_bn     = common::AutoSimplify(input_shape.back());
  int oc_bn_size = oc_bn.as_int32();
  int ic_bn_size = ic_bn.as_int32();
  VLOG(4) << "ow_bn_size " << ow_bn_size;
  VLOG(4) << "oc_bn_size " << oc_bn_size;
  VLOG(4) << "ic_bn_size " << ic_bn_size;

  // data
  if (data.defined()) {
    stages[data]->ComputeInline();
  }
  // weights
  if (weights_dilation.defined()) {
    CHECK_GE(stages[weights_dilation]->n_out_dims(), 3U) << "weights_dilation's out_dims should be more than 3";
    // Reorder: [oc_outer, ic_outer, oh, ow, ic_inner, oc_inner] ->
    // [oc_outer, oh, ic_outer, ow, ic_inner, oc_inner]
    stages[weights_dilation]->Reorder({2, 1});
  }
  // packed_out
  auto CC = stages[packed_out]->CacheWrite("global", stages, packed_out);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // packed_out: [batch, oc_outer, oh, ow, oc_inner]
  // split ow
  stages[packed_out]->Split(3, ow_bn_size);
  stages[packed_out]->Vectorize(stages[packed_out]->n_out_dims() - 1, packed_out->shape.back().as_int32());

  // CC: [batch, oc_outer, oh, ow, oc_inner]
  // packed_out: [batch, oc_outer, oh, ow_outer, ow_inner, oc_inner]
  stages[CC]->ComputeAt2(stages[packed_out], 3);
  VLOG(4) << "stages[packed_out]->transformed_domain()" << stages[packed_out]->transformed_domain();
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();

  // CC: [batch, oc_outer, oh, ow_outer, ow_inner, oc_inner, fc, kh, kw]
  // batch, oc_outer, oh, ow_outer, kh, kw, ow_inner, oc_inner
  auto CC_ow_inner = stages[CC]->axis(4);
  auto CC_oc_inner = stages[CC]->axis(5);
  auto CC_fc       = stages[CC]->axis(6);
  auto CC_kh       = stages[CC]->axis(7);
  auto CC_kw       = stages[CC]->axis(8);
  stages[CC]->Reorder({CC_fc, CC_kh, CC_kw, CC_ow_inner, CC_oc_inner});
  stages[CC]->Vectorize(stages[CC]->n_out_dims() - 1, CC->shape.back().as_int32());
  VLOG(4) << "stages[CC]->transformed_domain()" << stages[CC]->transformed_domain();
  // CC_init
  auto CC_init = CC->GetInitTensor(stages, target);
  stages[CC_init]->Vectorize(stages[CC_init]->n_out_dims() - 1, CC_init->shape.back().as_int32());

  // res
  // n, oc, oh, ow
  if (res.defined()) {
    stages[res]->Split(1, oc_bn_size);
    stages[res]->Split(4, ow_bn_size);
    // Reorder: [n, oc_outer, oc_inner, oh, ow_outer, ow_inner] ->
    // [n, oc_outer, oh, ow_outer, ow_inner, oc_inner]
    auto oc_inner1 = stages[res]->axis(2);
    auto oh1       = stages[res]->axis(3);
    auto ow_outer1 = stages[res]->axis(4);
    auto ow_inner1 = stages[res]->axis(5);
    stages[res]->Reorder({oh1, ow_outer1, ow_inner1, oc_inner1});
    VLOG(4) << "stages[res]->transformed_domain()" << stages[res]->transformed_domain();
  }
}

void CudaScheduleMul(poly::StageMap stages,
                     ir::Tensor output,
                     const std::vector<int> &output_shape,
                     const common::Target &target) {
  stages[output]->Split(1, 2);
  stages[output]->Bind(0, "blockIdx.x");
  stages[output]->Bind(1, "threadIdx.x");
}

inline void InputParam(std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> &model_data,
                       const std::string &key,
                       const std::vector<std::vector<int>> &int_data) {
  std::unordered_map<std::string, std::vector<int>> schedule_data;
  schedule_data["rc"] = int_data[0];
  schedule_data["ry"] = int_data[1];
  schedule_data["rx"] = int_data[2];
  schedule_data["f"]  = int_data[3];
  schedule_data["y"]  = int_data[4];
  schedule_data["x"]  = int_data[5];
  model_data[key]     = schedule_data;
}

void CreateSerialData(const std::string &file_name) {
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> model_data;
  // The format of serial data is:
  // hash_key: string = name of schedule + shape of input_pad + shape of weights + shape of output
  // value: vector of params
  InputParam(model_data,
             "CudaScheduleConv 1 3 230 230 64 3 7 7 1 64 112 112",
             {{3, 1}, {7, 1}, {1, 7}, {1, 4, 8, 2}, {112, 1, 1, 1}, {1, 7, 16, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 64 56 56 64 64 1 1 1 64 56 56",
             {{4, 16}, {1, 1}, {1, 1}, {1, 8, 8, 1}, {56, 1, 1, 1}, {1, 2, 28, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 64 58 58 128 64 3 3 1 128 28 28",
             {{32, 2}, {1, 3}, {1, 3}, {4, 2, 16, 1}, {28, 1, 1, 1}, {1, 2, 14, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 64 56 56 128 64 1 1 1 128 28 28",
             {{4, 16}, {1, 1}, {1, 1}, {2, 2, 32, 1}, {28, 1, 1, 1}, {1, 2, 14, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 128 30 30 256 128 3 3 1 256 14 14",
             {{32, 4}, {1, 3}, {1, 3}, {8, 1, 16, 2}, {7, 1, 2, 1}, {1, 1, 7, 2}});
  InputParam(model_data,
             "CudaScheduleConv 1 128 28 28 256 128 1 1 1 256 14 14",
             {{16, 8}, {1, 1}, {1, 1}, {8, 1, 16, 2}, {14, 1, 1, 1}, {1, 1, 14, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 256 16 16 512 256 3 3 1 512 7 7",
             {{64, 4}, {1, 3}, {1, 3}, {32, 1, 16, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});
  InputParam(model_data,
             "CudaScheduleConv 1 256 14 14 512 256 1 1 1 512 7 7",
             {{16, 16}, {1, 1}, {1, 1}, {16, 1, 32, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});

  // winograd
  InputParam(model_data,
             "CudaScheduleConv 1 64 58 58 64 64 3 3 1 64 56 56",
             {{32, 2}, {1, 3}, {1, 3}, {8, 1, 8, 1}, {28, 1, 2, 1}, {1, 4, 14, 1}});
  // winograd
  InputParam(model_data,
             "CudaScheduleConv 1 512 9 9 512 512 3 3 1 512 7 7",
             {{64, 8}, {1, 3}, {1, 3}, {32, 1, 16, 1}, {7, 1, 1, 1}, {1, 1, 7, 1}});
  // winograd
  InputParam(model_data,
             "CudaScheduleConv 1 256 16 16 256 256 3 3 1 256 14 14",
             {{64, 4}, {1, 3}, {1, 3}, {16, 1, 16, 1}, {14, 1, 1, 1}, {1, 1, 14, 1}});
  // winograd
  InputParam(model_data,
             "CudaScheduleConv 1 128 30 30 128 128 3 3 1 128 28 28",
             {{32, 4}, {1, 3}, {1, 3}, {8, 1, 16, 1}, {14, 1, 2, 1}, {1, 1, 7, 4}});

  proto::ModelData write_model_data;
  for (auto &i : model_data) {
    proto::ScheduleData write_schedule_data;
    for (auto &j : i.second) {
      proto::StringData write_vector_data;
      for (auto &k : j.second) {
        write_vector_data.add_data(std::to_string(k));
      }
      auto data_map        = write_schedule_data.mutable_data();
      (*data_map)[j.first] = write_vector_data;
    }
    auto model_map        = write_model_data.mutable_data();
    (*model_map)[i.first] = write_schedule_data;
    std::string test_write1;
    write_schedule_data.SerializeToString(&test_write1);
  }
  std::fstream output(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
  std::string test_write;
  write_model_data.SerializeToString(&test_write);
  if (!write_model_data.SerializeToOstream(&output)) {
    std::cerr << "Failed to write test_serial.log" << std::endl;
    exit(-1);
  }
  output.close();
}

int GetMaxSplitter(int a, int b) {
  while (a % b > 0) {
    b--;
  }
  return b;
}

void LoadSerialData(const std::string &file_name) {
  proto::ModelData read_model_data;
  std::fstream input(file_name, std::ios::in | std::ios::binary);
  if (!read_model_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse address book." << std::endl;
    exit(-1);
  }
  input.close();
  std::string test_write3;
  read_model_data.SerializeToString(&test_write3);
  auto read_model_map = read_model_data.data();
  auto &res           = ScheduleParam::get_instance().GetParam();
  for (auto &i : read_model_map) {
    auto read_schedule_map = i.second.data();
    std::unordered_map<std::string, std::vector<int>> param_data;
    for (auto &j : read_schedule_map) {
      std::vector<int> temp_data;
      for (int k = 0; k < j.second.data_size(); k++) {
        temp_data.push_back(std::stoi(j.second.data(k)));
      }
      param_data[j.first] = temp_data;
    }
    res[i.first] = param_data;
  }
}

void CudaScheduleConv(poly::StageMap stages,
                      ir::Tensor &input_pad,
                      ir::Tensor &weights,
                      ir::Tensor &output,
                      const common::Target &target) {
  auto &res = ScheduleParam::get_instance().GetParam();
  if (res.empty()) {
    CreateSerialData();
    LoadSerialData();
  }

  int n = output->shape[0].as_int32();
  int c = output->shape[1].as_int32();
  optim::Simplify(&(output->shape[2]));
  int h = output->shape[2].as_int32();
  optim::Simplify(&(output->shape[3]));
  int w  = output->shape[3].as_int32();
  int rc = input_pad->shape[1].as_int32();

  std::string key =
      "CudaScheduleConv " + std::to_string(input_pad->shape[0].as_int32()) + " " +
      std::to_string(input_pad->shape[1].as_int32()) + " " + std::to_string(input_pad->shape[2].as_int32()) + " " +
      std::to_string(input_pad->shape[3].as_int32()) + " " + std::to_string(weights->shape[0].as_int32()) + " " +
      std::to_string(weights->shape[1].as_int32()) + " " + std::to_string(weights->shape[2].as_int32()) + " " +
      std::to_string(weights->shape[3].as_int32()) + " " + std::to_string(output->shape[0].as_int32()) + " " +
      std::to_string(output->shape[1].as_int32()) + " " + std::to_string(output->shape[2].as_int32()) + " " +
      std::to_string(output->shape[3].as_int32());
  if (res.count(key) == 0) {
    LOG(INFO) << "Didn't find saved param, key is: " << key;
  } else {
    LOG(INFO) << "Find saved param! key is: " << key;
    CudaScheduleConv2(stages, input_pad, weights, output, target, key);
    return;
  }
  stages[input_pad]->ComputeInline();
  int f_inner  = GetInnerSplitter(c, h);
  int block_z  = SplitEven(c / f_inner);
  int thread_z = c / f_inner / block_z;

  int rc_factor = SplitEven(rc);

  auto OL = stages[output]->CacheWrite("local", stages, output);

  auto tx = stages[output]->axis(3);
  auto by = stages[output]->axis(2);
  auto[tem, fi] = stages[output]->Split(1, f_inner);
  auto[bz, tz]  = stages[output]->Split(1, thread_z);
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
  stages[OL]->Split(0, rc_factor);
  stages[OL]->Reorder({0, 2, 3, 1});
  stages[OL]->Bind(5, "blockIdx.z");
  stages[OL]->Bind(6, "blockIdx.y");
  stages[OL]->Bind(7, "threadIdx.z");
  stages[OL]->Bind(8, "threadIdx.x");
}

void CudaScheduleConv2(poly::StageMap stages,
                       ir::Tensor &input_pad,
                       ir::Tensor &weights,
                       ir::Tensor &output,
                       const common::Target &target,
                       const std::string &key) {
  auto &res = ScheduleParam::get_instance().GetParam();
  stages[input_pad]->ComputeInline();
  optim::Simplify(&(output->shape[2]));
  optim::Simplify(&(output->shape[3]));

  std::vector<ir::Tensor> readers{output};
  auto PR = stages[input_pad]->CacheRead("shared", readers, stages);
  auto KR = stages[weights]->CacheRead("shared", readers, stages);
  auto OL = stages[output]->CacheWrite("local", stages, output);

  auto &x_param  = res[key]["x"];
  auto &y_param  = res[key]["y"];
  auto &f_param  = res[key]["f"];
  auto &rx_param = res[key]["rx"];
  auto &ry_param = res[key]["ry"];
  auto &rc_param = res[key]["rc"];

  // x param is :  [1, 7, 16, 1]
  stages[output]->Split(3, x_param[3]);
  stages[output]->Split(3, x_param[2]);
  stages[output]->Split(3, x_param[1]);

  // y param is :  [112, 1, 1, 1]
  stages[output]->Split(2, y_param[3]);
  stages[output]->Split(2, y_param[2]);
  stages[output]->Split(2, y_param[1]);

  // f param is :  [1, 4, 8, 2]
  stages[output]->Split(1, f_param[3]);
  stages[output]->Split(1, f_param[2]);
  stages[output]->Split(1, f_param[1]);

  stages[output]->Reorder({0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  stages[output]->Bind(1, "blockIdx.z");
  stages[output]->Bind(2, "blockIdx.y");
  stages[output]->Bind(3, "blockIdx.x");
  stages[output]->Bind(7, "threadIdx.z");
  stages[output]->Bind(8, "threadIdx.y");
  stages[output]->Bind(9, "threadIdx.x");

  stages[OL]->ComputeAt(stages[output], 9);

  // rx param is :  [1, 7]
  stages[OL]->Split(15, rx_param[1]);
  // ry param is :  [7, 1]
  stages[OL]->Split(14, ry_param[1]);
  // rc param is :  [3, 1]
  stages[OL]->Split(13, rc_param[1]);

  stages[OL]->Reorder({13, 15, 17, 14, 16, 18, 10, 11, 12});

  auto OL_init = OL->GetInitTensor(stages, target);
  stages[PR]->ComputeAt(stages[OL], 12);
  stages[KR]->ComputeAt(stages[OL], 12);

  stages[PR]->SyncThreads(12, {OL_init}, stages);
  stages[KR]->CtrlDepend(PR);
  stages[KR]->SyncThreads(stages);

  if (stages[PR]->n_out_dims() == 18) {
    stages[PR]->Fuse({13, 14, 15, 16, 17});
  } else if (stages[PR]->n_out_dims() == 19) {
    stages[PR]->Fuse({13, 14, 15, 16, 17, 18});
  } else {
    LOG(FATAL) << "PR number of output dims is wrong: " << stages[PR]->n_out_dims();
  }

  if (stages[KR]->n_out_dims() == 18) {
    stages[KR]->Fuse({13, 14, 15, 16, 17});
  } else if (stages[KR]->n_out_dims() == 19) {
    stages[KR]->Fuse({13, 14, 15, 16, 17, 18});
  } else {
    LOG(FATAL) << "KR number of output dims is wrong: " << stages[KR]->n_out_dims();
  }
  int thread_z = f_param[2];
  int thread_x = x_param[2];
  if (stages[PR]->GetDimRange(13) <= thread_z) {
    stages[PR]->Bind(13, "threadIdx.z");
  } else {
    stages[PR]->Split(13, GetMaxSplitter(stages[PR]->GetDimRange(13), thread_z));
    stages[PR]->Bind(14, "threadIdx.z");
  }
  if (stages[KR]->GetDimRange(13) <= thread_x) {
    stages[KR]->Bind(13, "threadIdx.x");
  } else {
    stages[KR]->Split(13, GetMaxSplitter(stages[KR]->GetDimRange(13), thread_x));
    stages[KR]->Bind(14, "threadIdx.x");
  }
}

void CudaScheduleWinogradConv(poly::StageMap wino_stages,
                              ir::Tensor &wino_weights_dilation,
                              ir::Tensor &wino_input_pad,
                              ir::Tensor &wino_A,
                              ir::Tensor &wino_B,
                              ir::Tensor &wino_G,
                              ir::Tensor &kernel_pack,
                              ir::Tensor &input_tile,
                              ir::Tensor &data_pack,
                              ir::Tensor &bgemm,
                              ir::Tensor &inverse,
                              ir::Tensor &wino_conv,
                              const common::Target &target) {
  // s[B].compute_inline();
  wino_stages[wino_B]->ComputeInline();

  // data_l = s.cache_write(data_pack, "local")
  // eps, nu, c, p = s[data_l].op.axis
  // r_a, r_b = s[data_l].op.reduce_axis
  // for axis in [eps, nu, r_a, r_b]:
  //     s[data_l].unroll(axis)
  auto data_l = wino_stages[data_pack]->CacheWrite("local", wino_stages, data_pack);
  wino_stages[data_l]->Unroll(0);
  wino_stages[data_l]->Unroll(1);
  wino_stages[data_l]->Unroll(4);
  wino_stages[data_l]->Unroll(5);

  // eps, nu, c, p = s[data_pack].op.axis
  // p, pi = s[data_pack].split(p, 1)
  // fused = s[data_pack].fuse(c, p)
  // bb, tt = s[data_pack].split(fused, 128)
  // s[data_pack].reorder(bb, tt, pi, eps, nu)
  // s[data_pack].bind(bb, te.thread_axis("blockIdx.x"))
  // s[data_pack].bind(tt, te.thread_axis("threadIdx.x"))
  wino_stages[data_pack]->Fuse({2, 3});
  wino_stages[data_pack]->Split(2, 128);
  wino_stages[data_pack]->Reorder({2, 3, 0, 1});
  // wino_stages[data_pack]->Bind(0, "blockIdx.x");
  wino_stages[data_pack]->Bind(1, "threadIdx.x");

  // s[data_l].compute_at(s[data_pack], pi)
  // s[input_tile].compute_at(s[data_pack], pi)
  // s[pad_data].compute_inline()
  wino_stages[data_l]->ComputeAt(wino_stages[data_pack], 1);
  wino_stages[input_tile]->ComputeAt(wino_stages[data_l], 1);
  // wino_stages[input_tile]->SetBuffer("local");
  // wino_stages[input_tile]->ComputeInline();
  wino_stages[wino_input_pad]->ComputeInline();

  // kernel, G = s[kernel_pack].op.input_tensors
  // eps, nu, ci, co = s[kernel_pack].op.axis
  // s[G].compute_inline()
  // r_a, r_b = s[kernel_pack].op.reduce_axis
  // for axis in [eps, nu, r_a, r_b]:
  //     s[kernel_pack].unroll(axis)
  // fused = s[kernel_pack].fuse(ci, co)
  // bb, tt = s[kernel_pack].split(fused, 128)
  // s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b)
  // s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
  // s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))
  wino_stages[wino_G]->ComputeInline();
  wino_stages[kernel_pack]->Reorder({2, 3, 0, 1, 4, 5});
  wino_stages[kernel_pack]->Fuse({0, 1});
  wino_stages[kernel_pack]->Split(0, 128);
  wino_stages[kernel_pack]->Unroll(5);
  wino_stages[kernel_pack]->Unroll(4);
  wino_stages[kernel_pack]->Unroll(3);
  wino_stages[kernel_pack]->Unroll(2);
  wino_stages[kernel_pack]->Bind(0, "blockIdx.x");
  wino_stages[kernel_pack]->Bind(1, "threadIdx.x");

  // s[kernel].compute_inline()
  wino_stages[wino_weights_dilation]->ComputeInline();

  // C = bgemm
  // A0, B0 = kernel_pack, data_pack
  // OL = s.cache_write(C, "local")
  // AA = s.cache_read(A0, "shared", [OL])
  // BB = s.cache_read(B0, "shared", [OL])
  // std::vector<ir::Tensor> wino_readers{bgemm};
  // auto AA = wino_stages[kernel_pack]->CacheRead("shared", wino_readers, wino_stages);
  // auto BB = wino_stages[data_pack]->CacheRead("shared", wino_readers, wino_stages);
  auto wino_OL = wino_stages[bgemm]->CacheWrite("local", wino_stages, bgemm);

  // b1, b2, y, x = s[bgemm].op.axis
  // rc = s[bgemm].op.reduce_axis[0]
  // alpha = get_const_int(b1.dom.extent)
  // b = s[bgemm].fuse(b1, b2)
  // # tile and bind spatial axes
  // bgemm_scope, b = s[bgemm].split(b, nparts=1)
  // bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
  // by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
  // bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
  // s[C].bind(bz, te.thread_axis("blockIdx.z"))
  // s[C].bind(by, te.thread_axis("blockIdx.y"))
  // s[C].bind(bx, te.thread_axis("blockIdx.x"))
  // s[C].bind(vz, te.thread_axis("vthread"))
  // s[C].bind(vy, te.thread_axis("vthread"))
  // s[C].bind(vx, te.thread_axis("vthread"))
  // s[C].bind(tz, te.thread_axis("threadIdx.z"))
  // s[C].bind(ty, te.thread_axis("threadIdx.y"))
  // s[C].bind(tx, te.thread_axis("threadIdx.x"))
  // s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)
  wino_stages[bgemm]->Fuse({0, 1});

  // x param is :  [1, 2, 98, 1]
  wino_stages[bgemm]->Split(2, 1);
  wino_stages[bgemm]->Split(2, 98);
  wino_stages[bgemm]->Split(2, 2);
  // y param is :  [2, 2, 2, 8]
  wino_stages[bgemm]->Split(1, 8);
  wino_stages[bgemm]->Split(1, 2);
  wino_stages[bgemm]->Split(1, 2);
  // b param is :  [36, 1, 1, 1]
  wino_stages[bgemm]->Split(0, 1);
  wino_stages[bgemm]->Split(0, 1);
  wino_stages[bgemm]->Split(0, 1);

  wino_stages[bgemm]->Reorder({0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11});
  // wino_stages[bgemm]->Bind(0, "blockIdx.z");
  // wino_stages[bgemm]->Bind(1, "blockIdx.y");
  // wino_stages[bgemm]->Bind(2, "blockIdx.x");
  // wino_stages[bgemm]->Bind(6, "threadIdx.z");
  // wino_stages[bgemm]->Bind(7, "threadIdx.y");
  // wino_stages[bgemm]->Bind(8, "threadIdx.x");
  // wino_stages[bgemm]->SyncThreads(9, {wino_OL}, wino_stages);
  // wino_stages[bgemm]->SyncThreads(9, {}, wino_stages);
  // wino_stages[bgemm]->Bind(0, "blockIdx.x");
  wino_stages[bgemm]->Bind(8, "threadIdx.x");

  // # tile reduction axes
  // s[OL].compute_at(s[C], tx)
  // b1, b2, y, x = s[OL].op.axis
  // b = s[OL].fuse(b1, b2)
  // (rc,) = s[OL].op.reduce_axis
  // rco, rci = cfg["tile_rc"].apply(s, OL, rc)
  // s[OL].reorder(rco, rci, b, y, x)
  // s[AA].compute_at(s[OL], rco)
  // s[BB].compute_at(s[OL], rco)
  // LOG(INFO) << "bgemm";
  // wino_stages[bgemm]->ShowISL();
  // LOG(INFO) << "wino_OL before ";
  // wino_stages[wino_OL]->ShowISL();
  wino_stages[wino_OL]->ComputeAt(wino_stages[bgemm], 8);
  // LOG(INFO) << "wino_OL after ";
  // wino_stages[wino_OL]->ShowISL();
  // wino_stages[bgemm]->ShowISL();
  wino_stages[wino_OL]->Fuse({9, 10});
  // rc param is :  [8, 8]
  wino_stages[wino_OL]->Split(10, 8);
  wino_stages[wino_OL]->Reorder({10, 11, 9});
  // wino_stages[AA]->ComputeAt(wino_stages[wino_OL], 8);
  // wino_stages[BB]->ComputeAt(wino_stages[wino_OL], 8);

  // for load in [AA, BB]:
  //   fused = s[load].fuse(*list(s[load].op.axis))
  //   fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
  //   fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
  //   fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
  //   s[load].bind(tz, te.thread_axis("threadIdx.z"))
  //   s[load].bind(ty, te.thread_axis("threadIdx.y"))
  //   s[load].bind(tx, te.thread_axis("threadIdx.x"))
  // wino_stages[AA]->ShowISL();
  // wino_stages[AA]->Fuse({8, 9, 10, 11});
  // wino_stages[AA]->Split(8,20);
  // wino_stages[AA]->Split(8,4);
  // wino_stages[AA]->Split(8,2);
  // wino_stages[AA]->Bind(9, "blockIdx.z");
  // wino_stages[AA]->Bind(10, "blockIdx.y");
  // wino_stages[AA]->Bind(11, "blockIdx.x");

  // wino_stages[BB]->Fuse({8, 9, 10, 11});
  // wino_stages[BB]->Split(8,20);
  // wino_stages[BB]->Split(8,4);
  // wino_stages[BB]->Split(8,2);
  // wino_stages[BB]->Bind(9, "blockIdx.z");
  // wino_stages[BB]->Bind(10, "blockIdx.y");
  // wino_stages[BB]->Bind(11, "blockIdx.x");

  // m = alpha - 3 + 1
  // n, co, h, w = s[output].op.axis
  // ho, wo, hi, wi = s[output].tile(h, w, m, m)
  // inverse_scope, n = s[output].split(n, nparts=1)
  // fused = s[output].fuse(n, co, ho, wo)
  // bb, tt = s[output].split(fused, 128)
  // s[output].bind(bb, te.thread_axis("blockIdx.x"))
  // s[output].bind(tt, te.thread_axis("threadIdx.x"))
  int m = 4;
  wino_stages[wino_conv]->Tile(2, 3, m, m);
  // wino_stages[wino_conv]->Reorder({0, 1, 2, 4, 3, 5});
  wino_stages[wino_conv]->Fuse({0, 1, 2, 3});
  wino_stages[wino_conv]->Split(0, 128);
  // wino_stages[wino_conv]->Bind(0, "blockIdx.x");
  wino_stages[wino_conv]->Bind(1, "threadIdx.x");

  // s[A].compute_inline()
  wino_stages[wino_A]->ComputeInline();
  // LOG(INFO) << "inverse before ";
  // wino_stages[inverse]->ShowISL();
  // co, p, vh, vw = s[inverse].op.axis
  // r_a, r_b = s[inverse].op.reduce_axis
  // for axis in [vh, vw, r_a, r_b]:
  //     s[inverse].unroll(axis)
  // s[inverse].compute_at(s[output], tt)
  // wino_stages[inverse]->Unroll(2);
  // wino_stages[inverse]->Unroll(3);
  // wino_stages[inverse]->Unroll(4);
  // wino_stages[inverse]->Unroll(5);
  // wino_stages[inverse]->ComputeAt(wino_stages[wino_conv], 1);
  wino_stages[inverse]->Bind(1, "threadIdx.x");
  // wino_stages[wino_conv]->SyncThreads(1, {inverse}, wino_stages);
  LOG(INFO) << "inverse after ";
  wino_stages[inverse]->ShowISL();
  auto names = wino_stages[inverse]->axis_names();
  for (auto x : names) {
    LOG(INFO) << x;
  }
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
    auto[X_outer, X_inner]  = stage->Split(0, num_thread * num_block);
    auto[Block_x, Thread_x] = stage->Split(X_inner, num_thread);
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
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

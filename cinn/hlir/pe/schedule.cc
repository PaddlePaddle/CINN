#include "cinn/hlir/pe/schedule.h"

#include <isl/cpp.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "cinn/common/cas.h"
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

void CudaScheduleConv(poly::StageMap stages, ir::Tensor &input_pad, ir::Tensor &output, const common::Target &target) {
  stages[input_pad]->ComputeInline();
  int n = output->shape[0].as_int32();
  int c = output->shape[1].as_int32();
  optim::Simplify(&(output->shape[2]));
  int h = output->shape[2].as_int32();
  optim::Simplify(&(output->shape[3]));
  int w  = output->shape[3].as_int32();
  int rc = input_pad->shape[1].as_int32();

  int f_inner  = GetInnerSplitter(c, h);
  int block_z  = SplitEven(c / f_inner);
  int thread_z = c / f_inner / block_z;

  int rc_factor = SplitEven(rc);

  auto OL = stages[output]->CacheWrite("local", stages, output);

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
  stages[OL]->Split(0, rc_factor);
  stages[OL]->Reorder({0, 2, 3, 1});
  stages[OL]->Bind(5, "blockIdx.z");
  stages[OL]->Bind(6, "blockIdx.y");
  stages[OL]->Bind(7, "threadIdx.z");
  stages[OL]->Bind(8, "threadIdx.x");
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

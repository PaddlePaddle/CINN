#include "cinn/hlir/pe/transform.h"

#include "cinn/common/cas.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

void GetMatmulNewShapes(const std::vector<std::vector<int>> &inputs_shape,
                        bool trans_a,
                        bool trans_b,
                        std::vector<int> *new_shape_A,
                        std::vector<int> *new_shape_B,
                        std::vector<int> *output_shape) {
  *new_shape_A      = inputs_shape[0];
  *new_shape_B      = inputs_shape[1];
  int a_dim         = inputs_shape[0].size();
  int b_dim         = inputs_shape[1].size();
  int batch_shape_A = 1;
  int batch_shape_B = 1;
  int max_dim       = std::max(a_dim, b_dim);

  // broadcast dims if tensor's dim is 1
  if (max_dim == 1 && inputs_shape[0][0] != inputs_shape[1][0]) {
    // A: [M], B: [N] -> A: [M, 1], B: [1, N]
    *new_shape_A = {inputs_shape[0][0], 1};
    *new_shape_B = {1, inputs_shape[1][0]};
    trans_a      = false;
    trans_b      = false;
  } else {
    // A: [K], B: [K] -> A: [1, K], B: [K, 1]
    if (a_dim == 1) {
      *new_shape_A = {1, inputs_shape[0][0]};
      trans_a      = false;
    }
    if (b_dim == 1) {
      *new_shape_B = {inputs_shape[1][0], 1};
      trans_b      = false;
    }
  }
  // flatten batch dims
  if (max_dim > 3) {
    CHECK_EQ(a_dim, b_dim) << "tensors' dimension should be same if one of them is more than 3";
    for (int i = 0; i < a_dim - 2; ++i) {
      batch_shape_A = batch_shape_A * inputs_shape[0][i];
      batch_shape_B = batch_shape_B * inputs_shape[1][i];
    }
    CHECK(batch_shape_A == batch_shape_B || batch_shape_A == 1 || batch_shape_B == 1)
        << "batch dimension doesn't match";
    *new_shape_A = {batch_shape_A, inputs_shape[0][a_dim - 2], inputs_shape[0].back()};
    *new_shape_B = {batch_shape_B, inputs_shape[1][b_dim - 2], inputs_shape[1].back()};
  }

  max_dim = std::max(new_shape_A->size(), new_shape_B->size());
  if (new_shape_A->size() == 3U && new_shape_B->size() == 3U) {
    // eliminate batch 1
    if (new_shape_A->front() == 1 && new_shape_B->front() == 1) {
      new_shape_A->erase(new_shape_A->begin());
      new_shape_B->erase(new_shape_B->begin());
    }
  } else if (max_dim == 3) {
    // broadcast to 3D
    if (new_shape_A->size() == 2U) {
      new_shape_A->insert(new_shape_A->begin(), 1);
    }
    if (new_shape_B->size() == 2U) {
      new_shape_B->insert(new_shape_B->begin(), 1);
    }
  }
  CHECK(new_shape_A->size() == 3U || new_shape_A->size() == 2U) << "new_shape_A's dim should be 2 or 3";
  CHECK(new_shape_B->size() == 3U || new_shape_B->size() == 2U) << "new_shape_B's dim should be 2 or 3";
  int x_width  = trans_a ? (*new_shape_A)[new_shape_A->size() - 2] : new_shape_A->back();
  int y_height = trans_b ? new_shape_B->back() : (*new_shape_B)[new_shape_B->size() - 2];
  CHECK_EQ(x_width, y_height) << "matrix multiplication requires x_width to be same with y_height";
  if (new_shape_A->size() == 3U) {
    CHECK_EQ(new_shape_A->front(), new_shape_B->front())
        << "tensor A and B's batch size should be same but current batch sizes are " << new_shape_A->front() << " and "
        << new_shape_B->front();
  }
  if (output_shape != nullptr) {
    int M = !trans_a ? (*new_shape_A)[new_shape_A->size() - 2] : new_shape_A->back();
    int N = !trans_b ? new_shape_B->back() : (*new_shape_B)[new_shape_B->size() - 2];
    if (new_shape_A->size() == 3U) {
      *output_shape = {new_shape_A->front()};
    }
    output_shape->push_back(M);
    output_shape->push_back(N);
  }
}

std::shared_ptr<OpStrategy> StrategyForMatMul(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const std::vector<std::vector<int>> &output_shapes,
                                              const Target &target) {
  framework::CINNCompute matmul_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Matmul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for Matmul compute\n";
    Expr A = a[0];
    Expr B = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto attr_store = attrs.attr_store;
    bool trans_a    = false;
    bool trans_b    = false;
    float alpha     = 1;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "trans_a") {
        trans_a = absl::get<bool>(iter.second);
      } else if (iter.first == "trans_b") {
        trans_b = absl::get<bool>(iter.second);
      } else if (iter.first == "alpha") {
        alpha = absl::get<float>(iter.second);
      } else {
        LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
      }
    }
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages   = CreateStages({tensor_A, tensor_B});
    ir::Tensor new_A;
    ir::Tensor new_B;
    std::vector<int> old_shape_A;
    std::vector<int> old_shape_B;
    for (auto &shape : tensor_A->shape) {
      old_shape_A.push_back(shape.as_int32());
    }
    for (auto &shape : tensor_B->shape) {
      old_shape_B.push_back(shape.as_int32());
    }
    CHECK(!old_shape_A.empty());
    CHECK(!old_shape_B.empty());
    std::vector<int> new_shape_A = old_shape_A;
    std::vector<int> new_shape_B = old_shape_B;
    GetMatmulNewShapes({old_shape_A, old_shape_B}, trans_a, trans_b, &new_shape_A, &new_shape_B, nullptr);
    std::vector<Expr> new_shape_A_e;
    std::vector<Expr> new_shape_B_e;
    for (int shape : new_shape_A) {
      new_shape_A_e.push_back(Expr(shape));
    }
    for (int shape : new_shape_B) {
      new_shape_B_e.push_back(Expr(shape));
    }
    VLOG(4) << "matmul new_shape_A: " << new_shape_A_e;
    VLOG(4) << "matmul new_shape_B: " << new_shape_B_e;

    new_A = tensor_A->Reshape(new_shape_A_e, stages);
    new_B = tensor_B->Reshape(new_shape_B_e, stages);
    std::vector<ir::Tensor> out;
    if (target.arch == Target::Arch::X86) {
#ifdef CINN_WITH_MKL_CBLAS
      out = pe::MatmulMKL(new_A, new_B, trans_a, trans_b, alpha, UniqName("MatmulMKL_output"), target);
#else
      out = pe::MatmulV2(new_A, new_B, trans_a, trans_b, alpha, UniqName("MatmulV2_output"), target);
#endif
    } else {
      out = pe::Matmul(new_A, new_B, trans_a, trans_b, alpha, UniqName("Matmul_output"));
    }
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    CHECK(!out_type.empty()) << "Output type of MatMul is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule matmul_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of matmul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    int arg_size           = arg_pack.size();
    CHECK(arg_size == 2UL || arg_size == 3UL || arg_size == 4UL);
    poly::StageMap stages = arg_pack.back();
    if (target.arch == Target::Arch::NVGPU) {
      Expr out = arg_pack[0];
      CHECK(out.as_tensor());
      stages[out.as_tensor_ref()]->Split(1, 2);
      stages[out.as_tensor_ref()]->Bind(0, "blockIdx.x");
      stages[out.as_tensor_ref()]->Bind(1, "threadIdx.x");
    } else if (target.arch == Target::Arch::X86) {
#ifdef CINN_WITH_MKL_CBLAS
      CHECK_EQ(arg_pack.size(), 3UL);
#else
      CHECK(arg_pack.size() == 3UL || arg_pack.size() == 4UL);
      Expr packedB  = arg_pack[arg_size - 2];
      Expr temp_out = arg_pack[arg_size - 3];
      CHECK(packedB.as_tensor());
      CHECK(temp_out.as_tensor());
      pe::MatmulScheduleCPU(stages, temp_out.as_tensor_ref(), packedB.as_tensor_ref(), target);
#endif
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(matmul_compute, matmul_schedule, "strategy.matmul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMatMul(const std::vector<std::vector<int>> &inputs_shape,
                                                  framework::NodeAttr &attrs,
                                                  const Target &target) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<int> output_shape;
  std::vector<int> new_shape_A;
  std::vector<int> new_shape_B;
  bool trans_a = false;
  bool trans_b = false;
  float alpha  = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "trans_a") {
      trans_a = absl::get<bool>(iter.second);
    } else if (iter.first == "trans_b") {
      trans_b = absl::get<bool>(iter.second);
    } else if (iter.first == "alpha") {
      alpha = absl::get<float>(iter.second);
    }
  }
  GetMatmulNewShapes(inputs_shape, trans_a, trans_b, &new_shape_A, &new_shape_B, &output_shape);
  CHECK(!output_shape.empty()) << "infer_shape for matmul turns out to be empty. Please check\n";
  std::vector<int> packedB_shape;
  int shape_B_size = new_shape_B.size();
  CHECK_GE(new_shape_A.size(), 2U) << "new_shape_A's size should be no less than two";
  CHECK_GE(new_shape_B.size(), 2U) << "new_shape_B's size should be no less than two";
  CHECK_GE(output_shape.size(), 2U) << "output shape for matmul should be no less than two";
  int k  = new_shape_A.back();
  int n  = output_shape.back();
  int bn = pe::GetArrayPackingFactor(n, Float(32), common::DefaultHostTarget());

  packedB_shape = {n / bn, k, bn};
  if (output_shape.size() > 2) {
    CHECK_EQ(new_shape_A.size(), output_shape.size());
    packedB_shape.insert(packedB_shape.begin(), new_shape_A.front());
  }
  std::vector<std::vector<int>> res{output_shape, output_shape, packedB_shape};
  return res;
}

std::vector<Type> InferDtypeForMatMul(const std::vector<Type> &inputs_type,
                                      const framework::NodeAttr &attrs,
                                      const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0], inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForReshape2(const framework::NodeAttr &attrs,
                                                const std::vector<ir::Tensor> &inputs,
                                                const std::vector<Type> &out_type,
                                                const std::vector<std::vector<int>> &output_shapes,
                                                const Target &target) {
  framework::CINNCompute reshape2_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Matmul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 1U) << "at least 1 input tensors for Reshape2 compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto attr_store = attrs.attr_store;
    std::vector<int> new_shape;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "shape") {
        new_shape = absl::get<std::vector<int>>(iter.second);
      } else {
        LOG(FATAL) << "Unsupported attr: " << iter.first << std::endl;
      }
    }
    auto tensor_A = A.as_tensor_ref();
    auto stages   = CreateStages({tensor_A});
    ir::Tensor out;
    out = pe::Reshape2(tensor_A, output_shapes.back(), UniqName("Reshape2_output"));
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Reshape2 is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule reshape2_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of reshape2 schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    int arg_size           = arg_pack.size();
    poly::StageMap stages  = arg_pack.back();
    Expr out               = arg_pack[0];
    CHECK(out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes.back(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes.back(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reshape2_compute, reshape2_schedule, "strategy.reshape2.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForReshape2(const std::vector<std::vector<int>> &inputs_shape,
                                                    framework::NodeAttr &attrs,
                                                    const Target &target) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<int> output_shape;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "shape") {
      output_shape = absl::get<std::vector<int>>(iter.second);
    }
  }
  int tensor_size = 1;
  for (auto i : inputs_shape[0]) tensor_size *= i;
  CHECK(!output_shape.empty()) << "infer_shape for reshape2 turns out to be empty. Please check\n";
  int flag_index = -1;
  for (int i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > 0) {
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape2: " << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == -1 && flag_index == -1) {
      flag_index = i;
    } else if (output_shape[i] == -1) {
      LOG(FATAL) << "More than one -1 in output_shape of op reshape2.";
    } else {
      LOG(FATAL) << "Unsupported output_shape " << output_shape[i];
    }
  }
  if (flag_index >= 0) output_shape[flag_index] = tensor_size;
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForReshape2(const std::vector<Type> &inputs_type,
                                        const framework::NodeAttr &attrs,
                                        const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForMul(const framework::NodeAttr &attrs,
                                           const std::vector<ir::Tensor> &inputs,
                                           const std::vector<Type> &out_type,
                                           const std::vector<std::vector<int>> &output_shapes,
                                           const Target &target) {
  framework::CINNCompute mul_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Mul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for Mul compute\n";
    Expr A = a[0];
    Expr B = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto attr_store    = attrs.attr_store;
    int x_num_col_dims = 1;
    int y_num_col_dims = 1;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "x_num_col_dims") {
        x_num_col_dims = absl::get<int>(iter.second);
      } else if (iter.first == "y_num_col_dims") {
        y_num_col_dims = absl::get<int>(iter.second);
      }
    }
    auto A_tensor = A.as_tensor_ref();
    auto B_tensor = B.as_tensor_ref();
    auto stages   = CreateStages({A_tensor, B_tensor});
    std::vector<Expr> output_shape;
    std::vector<Expr> new_shape_A;
    std::vector<Expr> new_shape_B;
    Expr flatten_shape_A(1);
    Expr flatten_shape_B(1);
    Expr reduce_shape_A(1);
    Expr reduce_shape_B(1);
    for (int i = 0; i < A_tensor->shape.size(); i++) {
      if (i < x_num_col_dims) {
        flatten_shape_A = flatten_shape_A * A_tensor->shape[i];
      } else {
        reduce_shape_A = reduce_shape_A * A_tensor->shape[i];
      }
    }
    // flatten to 2 dims, new_shape_A: [M, K]
    flatten_shape_A = common::AutoSimplify(flatten_shape_A);
    reduce_shape_A  = common::AutoSimplify(reduce_shape_A);
    new_shape_A.push_back(flatten_shape_A);
    new_shape_A.push_back(reduce_shape_A);

    for (int i = 0; i < B_tensor->shape.size(); i++) {
      if (i < y_num_col_dims) {
        flatten_shape_B = flatten_shape_B * B_tensor->shape[i];
      } else {
        reduce_shape_B = reduce_shape_B * B_tensor->shape[i];
      }
    }
    flatten_shape_B = common::AutoSimplify(flatten_shape_B);
    reduce_shape_B  = common::AutoSimplify(reduce_shape_B);
    CHECK(is_zero(reduce_shape_A - reduce_shape_B)) << "reduce_shape should be same after flattening";
    // flatten to 2 dims, new_shape_B: [N, K]
    new_shape_B.push_back(flatten_shape_B);
    new_shape_B.push_back(reduce_shape_B);

    Var axis_k(reduce_shape_A, UniqName("axis_k"));
    auto new_A = A_tensor->Reshape(new_shape_A, stages);
    auto new_B = B_tensor->Reshape(new_shape_B, stages);
    std::vector<ir::Tensor> out;
    if (target.arch == Target::Arch::X86) {
#ifdef CINN_WITH_MKL_CBLAS
      out = pe::MulMKL(new_A, new_B, UniqName("Mul_mkl_output"), target);
#else
      out = pe::MulBase(new_A, new_B, UniqName("Mul_output"), target);
#endif
    } else {
      out = pe::MulBase(new_A, new_B, UniqName("Mul_output"), target);
    }
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    CHECK(!out_type.empty()) << "Output type of Mul is empty! Please check.\n";

    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule mul_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of mul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK(arg_pack.size() == 2UL || arg_pack.size() == 3UL);
    Expr out              = arg_pack[0];
    poly::StageMap stages = arg_pack.back();
    CHECK(out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleMul(stages, out.as_tensor_ref(), output_shapes.back(), target);
    } else if (target.arch == Target::Arch::X86) {
      CHECK_EQ(arg_pack.size(), 3UL);
#ifndef CINN_WITH_MKL_CBLAS
      Expr reduce_first = arg_pack[1];
      CHECK(reduce_first.as_tensor());
      pe::MulScheduleCPU(stages, out.as_tensor_ref(), reduce_first.as_tensor_ref(), target);
#endif
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.mul.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForMulBias(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const std::vector<std::vector<int>> &output_shapes,
                                               const Target &target) {
  framework::CINNCompute mul_bias_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Mul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 3U) << "at least 2 input tensors for Mul compute\n";
    Expr A = a[0];
    Expr B = a[1];
    Expr C = a[2];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK(C.as_tensor());
    auto attr_store    = attrs.attr_store;
    int x_num_col_dims = 1;
    int y_num_col_dims = 1;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "x_num_col_dims") {
        x_num_col_dims = absl::get<int>(iter.second);
      } else if (iter.first == "y_num_col_dims") {
        y_num_col_dims = absl::get<int>(iter.second);
      } else {
        LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
      }
    }
    auto A_tensor = A.as_tensor_ref();
    auto B_tensor = B.as_tensor_ref();
    auto C_tensor = C.as_tensor_ref();
    auto stages   = CreateStages({A_tensor, B_tensor, C_tensor});
    std::vector<Expr> output_shape;
    std::vector<Expr> new_xshape;
    std::vector<Expr> new_yshape;
    Expr check_dim(1);
    for (int i = 0; i < A_tensor->shape.size(); i++) {
      if (i < x_num_col_dims) {
        output_shape.push_back(A_tensor->shape[i]);
        new_xshape.push_back(A_tensor->shape[i]);
      } else {
        check_dim = check_dim * A_tensor->shape[i];
      }
    }
    new_xshape.push_back(check_dim);

    for (int i = 0; i < B_tensor->shape.size(); i++) {
      if (i < y_num_col_dims) {
        output_shape.push_back(B_tensor->shape[i]);
        new_yshape.push_back(B_tensor->shape[i]);
      }
    }
    new_yshape.push_back(check_dim);
    Var axis_k(check_dim, UniqName("axis_k"));
    auto new_A = A_tensor->Reshape(new_xshape, stages);
    auto new_B = B_tensor->Reshape(new_yshape, stages);

    auto out = pe::MulBias(new_A, new_B, C_tensor, x_num_col_dims, output_shape, axis_k, UniqName("MulBias_output"));

    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    res.push_back(CINNValue(stages));
    CHECK(!out_type.empty()) << "Output type of MulBias is empty! Please check.\n";
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule mul_bias_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of mul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    Expr temp             = arg_pack[0];
    Expr out              = arg_pack[1];
    poly::StageMap stages = arg_pack[2];
    CHECK(out.as_tensor());
    CHECK(temp.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleMul(stages, temp.as_tensor_ref(), output_shapes.back(), target);
      pe::CudaScheduleMul(stages, out.as_tensor_ref(), output_shapes.back(), target);
      /* stages[Out.as_tensor_ref()]->Split(1, 2);
      stages[Out.as_tensor_ref()]->Bind(0, "blockIdx.x");
      stages[Out.as_tensor_ref()]->Bind(1, "threadIdx.x"); */
      // pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.back(),target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_bias_compute, mul_bias_schedule, "strategy.mulbias.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMul(const std::vector<std::vector<int>> &inputs_shape,
                                               framework::NodeAttr &attrs,
                                               const Target &target) {
  // CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK_GE(inputs_shape[0].size(), 2U) << "Input matrix X's dim should be >= 2! Please check.";
  CHECK_GE(inputs_shape[1].size(), 2U) << "Input matrix Y's dim should be >= 2! Please check.";

  std::vector<int> output_shape;
  int x_num_col_dims = 1;
  int y_num_col_dims = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "x_num_col_dims") {
      x_num_col_dims = absl::get<int>(iter.second);
    } else if (iter.first == "y_num_col_dims") {
      y_num_col_dims = absl::get<int>(iter.second);
    } else {
      LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
    }
  }
  int flatten_shape_A = 1;
  int flatten_shape_B = 1;
  int check_dim_x     = 1;
  int check_dim_y     = 1;
  for (int i = 0; i < inputs_shape[0].size(); i++) {
    if (i < x_num_col_dims) {
      flatten_shape_A *= inputs_shape[0][i];
    } else {
      check_dim_x = check_dim_x * inputs_shape[0][i];
    }
  }

  for (int i = 0; i < inputs_shape[1].size(); i++) {
    if (i < y_num_col_dims) {
      flatten_shape_B *= inputs_shape[1][i];
    } else {
      check_dim_y = check_dim_y * inputs_shape[1][i];
    }
  }
  CHECK_EQ(check_dim_x, check_dim_y) << "For matrix multiply: X * Y, second dim of X's shape :[" << check_dim_x
                                     << "] should be equal to first dim of Y's shape :[" << check_dim_y
                                     << "]! Please Check!";
  output_shape = {flatten_shape_A, flatten_shape_B};

  int reduce_factor           = pe::GetMulFactor(check_dim_x, Float(32), common::DefaultHostTarget());
  std::vector<int> temp_shape = {flatten_shape_A, flatten_shape_B, reduce_factor};

  std::vector<std::vector<int>> res{output_shape, temp_shape};
  return res;
}

std::vector<Type> InferDtypeForMul(const std::vector<Type> &inputs_type,
                                   const framework::NodeAttr &attrs,
                                   const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForMul(const std::vector<framework::shape_t> &input_shapes,
                                                        const std::vector<std::string> &input_layouts,
                                                        const framework::NodeAttr &attrs,
                                                        const Target &target) {
  CHECK_EQ(input_layouts.size(), 2U) << "The input's layouts size is not 2! Please check again.";
  CHECK_EQ(input_shapes.size(), 2U) << "mul should have 2 input shapes";
  std::vector<std::string> new_input_layouts = input_layouts;
  for (int i = 0; i < input_shapes.size(); i++) {
    if (input_shapes[i].size() > 4) {
      new_input_layouts[i] = "NCHW";
    }
  }

  return {{"", ""}, new_input_layouts};
}

std::vector<std::vector<int>> InferShapeForMulBias(const std::vector<std::vector<int>> &inputs_shape,
                                                   framework::NodeAttr &attrs,
                                                   const Target &target) {
  // CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK_GE(inputs_shape[0].size(), 2U) << "Input matrix X's dim should be >= 2! Please check.";
  CHECK_GE(inputs_shape[1].size(), 2U) << "Input matrix Y's dim should be >= 2! Please check.";

  std::vector<int> output_shape;
  int x_num_col_dims = 1;
  int y_num_col_dims = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "x_num_col_dims") {
      x_num_col_dims = absl::get<int>(iter.second);
    } else if (iter.first == "y_num_col_dims") {
      y_num_col_dims = absl::get<int>(iter.second);
    } else {
      LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
    }
  }
  int check_dim_x = 1;
  int check_dim_y = 1;
  for (int i = 0; i < inputs_shape[0].size(); i++) {
    if (i < x_num_col_dims) {
      output_shape.push_back(inputs_shape[0][i]);
    } else {
      check_dim_x = check_dim_x * inputs_shape[0][i];
    }
  }

  for (int i = 0; i < inputs_shape[1].size(); i++) {
    if (i < y_num_col_dims) {
      output_shape.push_back(inputs_shape[1][i]);
    } else {
      check_dim_y = check_dim_y * inputs_shape[1][i];
    }
  }
  CHECK_EQ(check_dim_x, check_dim_y) << "For matrix multiply: X * Y, second dim of X's shape :[" << check_dim_x
                                     << "] should be equal to first dim of Y's shape :[" << check_dim_y
                                     << "]! Please Check!";

  std::vector<std::vector<int>> res{output_shape, output_shape};
  return res;
}

std::vector<Type> InferDtypeForMulBias(const std::vector<Type> &inputs_type,
                                       const framework::NodeAttr &attrs,
                                       const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForLayoutTransform(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  framework::CINNCompute layout_transform_compute([=](lang::Args args, lang::RetValue *ret) {
    std::string src_layout;
    std::string dst_layout;
    if (attrs.attr_store.find("src_layout") != attrs.attr_store.end()) {
      src_layout = absl::get<std::string>(attrs.attr_store.at("src_layout"));
    }
    if (attrs.attr_store.find("dst_layout") != attrs.attr_store.end()) {
      dst_layout = absl::get<std::string>(attrs.attr_store.at("dst_layout"));
    }
    CHECK(!args.empty()) << "The input argument of layout_transform compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for layout_transform compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());

    auto out    = pe::LayoutTransform(A.as_tensor_ref(), src_layout, dst_layout, UniqName("layout_transform_output"));
    auto stages = CreateStages({A.as_tensor_ref()});
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res  = {CINNValue(out), CINNValue(stages)};
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule layout_transform_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of layout_transform schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr out              = arg_pack[0];
    poly::StageMap stages = arg_pack[1];
    CHECK(out.as_tensor());
    auto tensor_out = out.as_tensor_ref();
    std::vector<int> out_shape;
    for (auto shape : tensor_out->shape) {
      out_shape.push_back(shape.as_int32());
    }
    if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPUFuse(stages[tensor_out], out_shape, target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of layout_transform op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(layout_transform_compute, layout_transform_schedule, "strategy.layout_transform.x86", 1);
  } else {
    LOG(FATAL) << "layout_transform op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<shape_t> InferShapeForLayoutTransform(const std::vector<shape_t> &inputs_shape,
                                                  framework::NodeAttr &attrs,
                                                  const Target &target) {
  std::string src_layout;
  std::string dst_layout;
  if (attrs.attr_store.find("src_layout") != attrs.attr_store.end()) {
    src_layout = absl::get<std::string>(attrs.attr_store.at("src_layout"));
  }
  if (attrs.attr_store.find("dst_layout") != attrs.attr_store.end()) {
    dst_layout = absl::get<std::string>(attrs.attr_store.at("dst_layout"));
  }
  CHECK_EQ(inputs_shape.size(), 1UL);

  std::vector<Expr> input_shapes_expr;
  for (int shape : inputs_shape[0]) {
    input_shapes_expr.push_back(Expr(shape));
  }
  absl::flat_hash_map<int, std::vector<int>> split_index_map;
  std::vector<Expr> out_shapes = pe::InferShapeLayoutTransform(
      input_shapes_expr, ir::Layout(src_layout), ir::Layout(dst_layout), &split_index_map);
  VLOG(4) << "out_shapes: " << out_shapes;
  std::vector<int> output_shapes;
  for (auto &shape : out_shapes) {
    output_shapes.push_back(shape.as_int32());
  }
  return {output_shapes};
}

std::vector<Type> InferDtypeForLayoutTransform(const std::vector<Type> &inputs_type,
                                               const framework::NodeAttr &attrs,
                                               const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForLayoutTransform(const std::vector<framework::shape_t> &input_shapes,
                                                                    const std::vector<std::string> &input_layouts,
                                                                    const framework::NodeAttr &attrs,
                                                                    const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  std::string dst_layout;
  std::string src_layout;
  if (attrs.attr_store.find("dst_layout") != attrs.attr_store.end()) {
    dst_layout = absl::get<std::string>(attrs.attr_store.at("dst_layout"));
  }
  if (attrs.attr_store.find("src_layout") != attrs.attr_store.end()) {
    src_layout = absl::get<std::string>(attrs.attr_store.at("src_layout"));
  }
  return {{dst_layout}, {src_layout}};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(transform_ops) {
  CINN_REGISTER_OP(matmul)
      .describe(
          "This operator is used to perform (batched) matrix multiplication over the last two dimensions of the input "
          "tensors X and Y.")
      .set_num_inputs(2)
      .set_num_outputs(3)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMatMul)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForMatMul))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForMatMul))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  CINN_REGISTER_OP(reshape2)
      .describe("This operator is used to reshape input tensor X.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForReshape2)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForReshape2))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForReshape2))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  CINN_REGISTER_OP(mul)
      .describe("This operator is used to perform matrix multiplication for input X and Y.")
      .set_num_inputs(2)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMul)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForMul))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForMul))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForMul))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  CINN_REGISTER_OP(mulbias)
      .describe("This operator is used to perform matrix multiplication for input X and Y and add Z.")
      .set_num_inputs(3)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMulBias)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForMulBias))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForMulBias))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern",
                                                      cinn::hlir::framework::OpPatternKind::kOutEWiseFusable)
      .set_support_level(4);

  CINN_REGISTER_OP(layout_transform)
      .describe("This operator is used to transform op's layouts")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForLayoutTransform)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForLayoutTransform))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForLayoutTransform))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForLayoutTransform))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);
  return true;
}

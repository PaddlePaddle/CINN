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
        trans_a = std::get<bool>(iter.second);
      } else if (iter.first == "trans_b") {
        trans_b = std::get<bool>(iter.second);
      } else if (iter.first == "alpha") {
        alpha = std::get<int>(iter.second);
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
    out = pe::Matmul(new_A, new_B, trans_a, trans_b, alpha, UniqName("Matmul_output"));
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
    Expr out              = arg_pack[0];
    poly::StageMap stages = arg_pack.back();
    CHECK(out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      stages[out.as_tensor_ref()]->Split(1, 2);
      stages[out.as_tensor_ref()]->Bind(0, "blockIdx.x");
      stages[out.as_tensor_ref()]->Bind(1, "threadIdx.x");
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(matmul_compute, matmul_schedule, "strategy.matmul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForMatMul(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::NodeAttr &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<int> output_shape;
  std::vector<int> new_shape_A;
  std::vector<int> new_shape_B;
  bool trans_a = false;
  bool trans_b = false;
  float alpha  = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "trans_a") {
      trans_a = std::get<bool>(iter.second);
    } else if (iter.first == "trans_b") {
      trans_b = std::get<bool>(iter.second);
    } else if (iter.first == "alpha") {
      alpha = std::get<float>(iter.second);
    }
  }
  GetMatmulNewShapes(inputs_shape, trans_a, trans_b, &new_shape_A, &new_shape_B, &output_shape);
  CHECK(!output_shape.empty()) << "infer_shape for matmul turns out to be empty. Please check\n";
  std::vector<std::vector<int>> res{output_shape, output_shape};
  return res;
}

std::vector<Type> InferDtypeForMatMul(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
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
        x_num_col_dims = std::get<int>(iter.second);
      } else if (iter.first == "y_num_col_dims") {
        y_num_col_dims = std::get<int>(iter.second);
      } else {
        LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
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

    auto out = pe::MulBase(new_A, new_B, UniqName("Mul_output"), target);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    CHECK(!out_type.empty()) << "Output type of Mul is empty! Please check.\n";

    if (target.arch == Target::Arch::NVGPU) {
      std::vector<ir::Tensor> readers{out};
      auto BB = stages[new_B]->CacheRead2("local", readers, stages);
      stages[BB]->Split(0, 2);
      stages[BB]->Bind(0, "threadIdx.x");
    }
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
      Expr reduce_first = arg_pack[1];
      CHECK(reduce_first.as_tensor());
      pe::MulScheduleCPU(stages, out.as_tensor_ref(), reduce_first.as_tensor_ref(), target);
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
        x_num_col_dims = std::get<int>(iter.second);
      } else if (iter.first == "y_num_col_dims") {
        y_num_col_dims = std::get<int>(iter.second);
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
                                               const framework::NodeAttr &attrs) {
  // CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK_GE(inputs_shape[0].size(), 2U) << "Input matrix X's dim should be >= 2! Please check.";
  CHECK_GE(inputs_shape[1].size(), 2U) << "Input matrix Y's dim should be >= 2! Please check.";

  std::vector<int> output_shape;
  int x_num_col_dims = 1;
  int y_num_col_dims = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "x_num_col_dims") {
      x_num_col_dims = std::get<int>(iter.second);
    } else if (iter.first == "y_num_col_dims") {
      y_num_col_dims = std::get<int>(iter.second);
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

  int reduce_factor           = pe::GetMulReduceFactor(check_dim_x, Float(32), common::DefaultHostTarget());
  std::vector<int> temp_shape = {flatten_shape_A, flatten_shape_B, reduce_factor};

  std::vector<std::vector<int>> res{output_shape, temp_shape};
  return res;
}

std::vector<Type> InferDtypeForMul(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

std::vector<std::vector<int>> InferShapeForMulBias(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::NodeAttr &attrs) {
  // CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK_GE(inputs_shape[0].size(), 2U) << "Input matrix X's dim should be >= 2! Please check.";
  CHECK_GE(inputs_shape[1].size(), 2U) << "Input matrix Y's dim should be >= 2! Please check.";

  std::vector<int> output_shape;
  int x_num_col_dims = 1;
  int y_num_col_dims = 1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "x_num_col_dims") {
      x_num_col_dims = std::get<int>(iter.second);
    } else if (iter.first == "y_num_col_dims") {
      y_num_col_dims = std::get<int>(iter.second);
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

std::vector<Type> InferDtypeForMulBias(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
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
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMatMul)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForMatMul))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForMatMul))
      .set_support_level(4);

  CINN_REGISTER_OP(mul)
      .describe("This operator is used to perform matrix multiplication for input X and Y.")
      .set_num_inputs(2)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMul)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForMul))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForMul))
      .set_support_level(4);

  CINN_REGISTER_OP(mulbias)
      .describe("This operator is used to perform matrix multiplication for input X and Y and add Z.")
      .set_num_inputs(3)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForMulBias)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForMulBias))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForMulBias))
      .set_support_level(4);
  return true;
}

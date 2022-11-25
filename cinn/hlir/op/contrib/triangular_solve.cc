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

#include "cinn/hlir/op/contrib/triangular_solve.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
DECLARE_bool(cinn_ir_schedule);
#define PI 3.14159265357989

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor TriangularSolve(const ir::Tensor &A,
                           const ir::Tensor &B,
                           const bool &left_side,
                           const bool &upper,
                           const bool &transpose_a,
                           const bool &unit_diagonal,
                           const std::string &name,
                           const common::Target &target) {
  CHECK_EQ(A->shape.size(), B->shape.size());
  if (target.arch == Target::Arch::X86) {
    std::vector<Expr> shape_A = A->shape;
    std::vector<Expr> shape_B = B->shape;
    int a_dim                 = shape_A.size();
    int b_dim                 = shape_B.size();
    CHECK(a_dim == 3U || a_dim == 2U) << "tensor_A's dim should be 2 or 3 while current dim is " << a_dim;
    CHECK(b_dim == 3U || b_dim == 2U) << "tensor_B's dim should be 2 or 3 while current dim is " << b_dim;
    CHECK_EQ(a_dim, b_dim) << "tensor_A's dim should be same with tensor_B";
    if (a_dim == 3U) {
      CHECK_EQ(shape_A.front(), shape_B.front())
          << "tensor A and B's batch size should be same but current batch sizes are " << shape_A.front() << " and "
          << shape_B.front();
    }

    Expr M  = shape_A[a_dim - 2];
    Expr M2 = shape_A.back();
    Expr N  = shape_B.back();
    CHECK(is_zero(M - M2)) << "matrix A requires width to be same with height";

    ir::Tensor call;
    if (a_dim == 2U) {
      call = Compute(
          {Expr(1)},
          [=]() -> Expr {
            return lang::CallExtern("cinn_cpu_mkl_trsm_fp32",
                                    {
                                        Expr(1.0),                         // alpha
                                        M,                                 // M
                                        N,                                 // N
                                        common::make_bool(left_side),      // left_side
                                        common::make_bool(upper),          // upper
                                        common::make_bool(transpose_a),    // transpose_a
                                        common::make_bool(unit_diagonal),  // unit_diagonal
                                        shape_A.back(),                    // lda
                                        shape_B.back(),                    // ldb
                                        A,                                 // A
                                        B,                                 // B
                                    });
          },
          UniqName("TriangularSolve_mkl_out"));
    } else {
      // batch TriangularSolve
      call = Compute(
          {Expr(1)},
          [=]() -> Expr {
            return lang::CallExtern("cinn_cpu_mkl_trsm_batch_fp32",
                                    {
                                        Expr(1.0),                         // alpha
                                        shape_A.front(),                   // batch
                                        M,                                 // M
                                        N,                                 // N
                                        common::make_bool(left_side),      // left_side
                                        common::make_bool(upper),          // upper
                                        common::make_bool(transpose_a),    // transpose_a
                                        common::make_bool(unit_diagonal),  // unit_diagonal
                                        shape_A.back(),                    // lda
                                        shape_B.back(),                    // ldb
                                        M * M,                             // a_stride
                                        N * N,                             // b_stride
                                        A,                                 // A
                                        B,                                 // B
                                    });
          },
          UniqName("batch_TriangularSolve_mkl_out"));
    }
    auto out = call->TupleGet(0);
    out->WithBuffer(A->type());
    return out;
  } else {
  }
}

std::shared_ptr<OpStrategy> StrategyForTriangularSolve(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  framework::CINNCompute matmul_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Matmul compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U) << "at least 2 input tensors for Matmul compute\n";
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    auto attr_store    = attrs.attr_store;
    bool left_side     = false;
    bool upper         = false;
    bool transpose_a   = false;
    bool unit_diagonal = false;
    float alpha        = 1;
    if (attr_store.count("left_side")) {
      left_side = absl::get<bool>(attr_store.at("left_side"));
    }
    if (attr_store.count("upper")) {
      upper = absl::get<bool>(attr_store.at("upper"));
    }
    if (attr_store.count("transpose_a")) {
      transpose_a = absl::get<bool>(attr_store.at("transpose_a"));
    }
    if (attr_store.count("unit_diagonal")) {
      unit_diagonal = absl::get<bool>(attr_store.at("unit_diagonal"));
    }
    if (attr_store.count("alpha")) {
      alpha = absl::get<float>(attr_store.at("alpha"));
    }

    std::string tensor_name = UniqName("TriangularSolve");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_GE(pack_args.size(), 3);
      CHECK(pack_args[2].is_string());
      tensor_name = pack_args[2].operator std::string();
    }

    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages   = CreateStages({tensor_A, tensor_B});

    std::vector<ir::Tensor> out;
    out = TriangularSolve(
        tensor_A, tensor_B, left_side, upper, transpose_a, unit_diagona, UniqName("TriangularSolve_output"), target);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    CHECK(!out_type.empty()) << "Output type of TriangularSolve is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      matmul_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.triangular_solve.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForTriangularSolve(const std::vector<std::vector<int>> &inputs_shape,
                                                           const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[1]};
  return res;
}

std::vector<Type> InferDtypeForTriangularSolve(const std::vector<Type> &inputs_type,
                                               const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
#ifdef CINN_WITH_CUDA
  std::vector<Type> res{inputs_type[0]};
#else
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
#endif
  return res;
}

CINN_REGISTER_HELPER(triangular_solve_ops) {
  CINN_REGISTER_OP(triangular_solve)
      .describe("This operator implements the op TriangularSolve.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForTriangularSolve)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForTriangularSolve))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForTriangularSolve))
      .set_support_level(4);

  return true;
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

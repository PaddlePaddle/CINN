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

#include "cinn/hlir/op/contrib/cholesky.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "cinn/common/cas.h"
#include "cinn/common/cinn_value.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/macros.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/packed_func.h"
#include "cinn/poly/stage.h"
#include "glog/logging.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;
using framework::shape_t;

ir::Tensor Cholesky(const ir::Tensor &x,
                    const bool upper,
                    const Target &target,
                    const std::string &output_name) {
  std::string extern_func = "cinn_";
  if (target == common::DefaultHostTarget()) {
    extern_func += "cpu_mkl_";
  } else if (target == common::DefaultNVGPUTarget()) {
    // extern_func += "gpu_cusolver_";
    CINN_NOT_IMPLEMENTED
  } else {
    CINN_NOT_IMPLEMENTED
  }

  extern_func += "cholesky";

  if (x->type().is_float(32)) {
    extern_func += "_fp32";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  // 计算batch_size，即总共需要调用核函数的次数
  int ndim = static_cast<int>(x->shape.size());
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= x->shape[i].as_int32();
  }
  // 获取正定矩阵的维度M
  int m = x->shape[ndim - 1].as_int32();

  auto res = Compute(
    {Expr(1)},
    [=]() {
        return lang::CallExtern(extern_func,
                                {
                                  x,  // Input matrix
                                  Expr(batch_size),  // Batch size
                                  Expr(m),  // Matrix shape
                                  common::make_bool(upper)
                                });
    },
    output_name
  );
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForCholesky(const framework::NodeAttr &attrs,
                                                           const std::vector<ir::Tensor> &inputs,
                                                           const std::vector<Type> &out_type,
                                                           const std::vector<std::vector<int>> &output_shapes,
                                                           const Target &target) {
  std::string op_name = "cholesky";

  framework::CINNCompute cholesky_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "at least one input tensor for " << op_name << " compute.";

    auto attr_store = attrs.attr_store;
    bool upper = false;
    if (attr_store.count("upper")) {
        upper = absl::get<bool>(attr_store.at("upper"));
    }

    std::string tensor_name = UniqName(op_name + "_Out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }

    Expr x = pack_args[0];
    CHECK(x.as_tensor());
    ir::Tensor A = x.as_tensor_ref();
    auto out = Cholesky(A, upper, target, tensor_name);
    auto stages = CreateStages({out});
    *ret = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cholesky_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.cholesky.x86", 1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForCholesky(const std::vector<framework::shape_t> &inputs_shape,
                                                      const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  framework::shape_t x_shape = inputs_shape[0];
  CHECK_GE(x_shape.size(), 2U) << "The input x shape size should >= 2! Please check again.";
  CHECK_EQ(x_shape[x_shape.size() - 2], x_shape[x_shape.size() - 1]) << "The last two dimensions of the input x must be the same!";
  return inputs_shape;
}

std::vector<Type> InferDtypeForCholesky(const std::vector<Type> &inputs_type,
                                        const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn
// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "infrt/dialect/init_cinn_dialects.h"

#include <glog/logging.h>

#include "infrt/dialect/basic_kernels.h"
#include "infrt/dialect/cinn_base.h"
#include "infrt/dialect/dense_tensor.h"
#include "infrt/dialect/pd_ops.h"
#include "infrt/dialect/tensor_shape.h"

namespace infrt {

void RegisterCinnDialects(mlir::DialectRegistry& registry) {
  registry.insert<ts::TensorShapeDialect>();
  registry.insert<dialect::CINNDialect>();
  registry.insert<dt::DTDialect>();
  registry.insert<mlir::pd::PaddleDialect>();
}

}  // namespace infrt

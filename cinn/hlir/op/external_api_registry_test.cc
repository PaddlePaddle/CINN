// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/op/external_api_registry.h"

#include <gtest/gtest.h>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace hlir {
namespace framework {

using cinn::hlir::framework::Node;
using cinn::hlir::op::ExternalApiRegistry;

TEST(ExternalApiRegistry, Has) {
  ASSERT_TRUE(ExternalApiRegistry::Global()->Has("matmul", common::DefaultNVGPUTarget()));
  ASSERT_TRUE(ExternalApiRegistry::Global()->Has("cholesky", common::DefaultHostTarget()));
  ASSERT_FALSE(ExternalApiRegistry::Global()->Has("op_doesn't_exist", common::DefaultNVGPUTarget()));
}

TEST(ExternalApiRegistry, GetExternalApi) {
  auto matmul_node = std::make_unique<Node>(Operator::Get("matmul"), "matmul");
  ASSERT_EQ("cinn_call_cublas",
            ExternalApiRegistry::Global()->GetExternalApi(matmul_node.get(), common::DefaultNVGPUTarget()));
  auto conv2d_node                           = std::make_unique<Node>(Operator::Get("conv2d"), "conv2d");
  conv2d_node->attrs.attr_store["conv_type"] = std::string("backward_data");
  ASSERT_EQ("cinn_call_cudnn_conv2d_backward_data",
            ExternalApiRegistry::Global()->GetExternalApi(conv2d_node.get(), common::DefaultNVGPUTarget()));
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

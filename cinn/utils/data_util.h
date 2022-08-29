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

#pragma once
#include <random>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/tensor.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
template <typename T>
void SetRandData(hlir::framework::Tensor tensor, const common::Target& target, int seed = -1);

template <typename T>
std::vector<T> GetTensorData(const hlir::framework::Tensor& tensor, const common::Target& target);

}  // namespace cinn

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
/**
 * \file This file implements some intrinsic functions for math operation in cuda device.
 */
#include "cinn/runtime/cinn_runtime.h"

extern "C" {

__device__ inline int cinn_cuda_find_int_nd(const int *buf, int size, int num, int begin, int stride);

__device__ inline int cinn_cuda_find_float_nd(const float *buf, int size, float num, int begin, int stride);
}

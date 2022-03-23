// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
 * \file This file implements some intrinsic functions for math operation in host device.
 */
#include "cinn/runtime/cinn_runtime.h"

extern "C" {

//! math extern functions
//@{
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out);
//@}

#define CINN_HOST_FIND_FOREACH_TYPE(MACRO) \
  MACRO(int)                               \
  MACRO(float)                             \
  MACRO(int64_t)

#define DECALARE_CINN_HOST_FIND(TYPE) inline int cinn_host_find_##TYPE(const cinn_buffer_t* buf, int size, int num);

CINN_HOST_FIND_FOREACH_TYPE(DECALARE_CINN_HOST_FIND)
#undef DECALARE_CINN_HOST_FIND
}

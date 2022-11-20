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

inline int cinn_host_find_int(const cinn_buffer_t* buf, int size, int num);

inline int cinn_host_find_float(const cinn_buffer_t* buf, int size, float num);

inline int cinn_host_find_int_nd(const cinn_buffer_t* buf, int size, int num, int begin, int stride);

inline int cinn_host_find_float_nd(const cinn_buffer_t* buf, int size, float num, int begin, int stride);

inline int cinn_host_lt_num_float(
    const cinn_buffer_t* buf, const int size, const float num, const int offset, const int stride);

inline int cinn_host_lt_num_int(
    const cinn_buffer_t* buf, const int size, const int num, const int offset, const int stride);

inline int cinn_host_gt_num_float(
    const cinn_buffer_t* buf, const int size, const float num, const int offset, const int stride);

inline int cinn_host_gt_num_int(
    const cinn_buffer_t* buf, const int size, const int num, const int offset, const int stride);

#define FN_INT32(func) cinn_host_##func##_int32

inline int FN_INT32(pow)(int x, int y);

inline int FN_INT32(clz)(int x);

#undef FN_INT32

#define FN_INT64(func) cinn_host_##func##_uint64

inline int64_t FN_INT64(clz)(int64_t x);

#undef FN_INT64
}

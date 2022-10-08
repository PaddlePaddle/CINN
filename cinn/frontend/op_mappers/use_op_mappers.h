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

#include "cinn/common/macros.h"

CINN_USE_REGISTER(paddle_sigmoid)
CINN_USE_REGISTER(paddle_fetch_feed)
CINN_USE_REGISTER(paddle_mul)
CINN_USE_REGISTER(paddle_slice)
CINN_USE_REGISTER(paddle_relu)
CINN_USE_REGISTER(paddle_softmax)
CINN_USE_REGISTER(paddle_scale)
CINN_USE_REGISTER(paddle_batchnorm)
CINN_USE_REGISTER(paddle_dropout)
CINN_USE_REGISTER(paddle_elementwise)
CINN_USE_REGISTER(paddle_pool2d)
CINN_USE_REGISTER(paddle_conv2d)
CINN_USE_REGISTER(paddle_transpose)
CINN_USE_REGISTER(paddle_reshape)
CINN_USE_REGISTER(paddle_tanh)
CINN_USE_REGISTER(paddle_matmul)
CINN_USE_REGISTER(paddle_compare)
CINN_USE_REGISTER(paddle_log)
CINN_USE_REGISTER(paddle_concat)
CINN_USE_REGISTER(paddle_constant)
CINN_USE_REGISTER(paddle_where)
CINN_USE_REGISTER(paddle_squeeze)
CINN_USE_REGISTER(paddle_expand)

CINN_USE_REGISTER(science_broadcast)
CINN_USE_REGISTER(science_transform)
CINN_USE_REGISTER(science_math)

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

#include <set>
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

static const std::set<std::string> kExternFp32CallsGPU{
    {"exp",         "erf",         "sigmoid",     "sqrt",        "log",        "log2",        "log10",
     "floor",       "ceil",        "round",       "trunc",       "cos",        "cosh",        "tan",
     "sin",         "sinh",        "acos",        "acosh",       "asin",       "asinh",       "atan",
     "atanh",       "isnan",       "tanh",        "isfinite",    "isinf",      "left_shift",  "right_shift",
     "bitwise_or",  "bitwise_and", "bitwise_xor", "bitwise_not", "left_shift", "right_shift", "bitwise_or",
     "bitwise_and", "bitwise_xor", "bitwise_not"}};

static const std::set<std::string> kExternFp32CallsCPU = {"erf", "acos", "acosh", "asin", "asinh", "atan", "atanh"};

/**
 * Map the Call nodes to external function call.
 *
 * This will rename the external call with the function in different backends.
 */
void MapExternCall(Expr *e, Target target);

}  // namespace optim
}  // namespace cinn

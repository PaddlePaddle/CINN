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

#include "cinn/ir/ir_constant.h"

#include <limits>

namespace cinn {
namespace ir {

Expr Zero(const Type& type) {
  if (type.is_float(32)) return Expr(0.f);
  if (type.is_float(64)) return Expr(double(0.));  // NOLINT
  if (type.is_bool()) return Expr(false);
  if (type.is_int(32)) return Expr(int32_t(0));
  if (type.is_int(64)) return Expr(int64_t(0));
  if (type.is_uint(32)) return Expr(uint32_t(0));
  if (type.is_uint(64)) return Expr(uint64_t(0));
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr One(const Type& type) {
  if (type.is_float(32)) return Expr(1.0f);
  if (type.is_float(64)) return Expr(double(1.0));  // NOLINT
  if (type.is_bool()) return Expr(true);
  if (type.is_int(32)) return Expr(int32_t(1));
  if (type.is_int(64)) return Expr(int64_t(1));
  if (type.is_uint(32)) return Expr(uint32_t(1));
  if (type.is_uint(64)) return Expr(uint64_t(1));
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr Maximum(const Type& type) {
  if (type.is_float(32)) return Expr(std::numeric_limits<float>::max());
  if (type.is_float(64)) return Expr(std::numeric_limits<double>::max());  // NOLINT
  if (type.is_bool()) return Expr(true);
  if (type.is_int(32)) return Expr(std::numeric_limits<int32_t>::max());
  if (type.is_int(64)) return Expr(std::numeric_limits<int64_t>::max());
  if (type.is_uint(32)) return Expr(std::numeric_limits<uint32_t>::max());
  if (type.is_uint(64)) return Expr(std::numeric_limits<uint64_t>::max());
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr Minimum(const Type& type) {
  if (type.is_float(32)) return Expr(std::numeric_limits<float>::min());
  if (type.is_float(64)) return Expr(std::numeric_limits<double>::min());  // NOLINT
  if (type.is_bool()) return Expr(false);
  if (type.is_int(32)) return Expr(std::numeric_limits<int32_t>::min());
  if (type.is_int(64)) return Expr(std::numeric_limits<int64_t>::min());
  if (type.is_uint(32)) return Expr(std::numeric_limits<uint32_t>::min());
  if (type.is_uint(64)) return Expr(std::numeric_limits<uint64_t>::min());
  CINN_NOT_IMPLEMENTED
  return Expr();
}

}  // namespace ir
}  // namespace cinn

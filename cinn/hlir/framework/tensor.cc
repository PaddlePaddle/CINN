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

#include "cinn/hlir/framework/tensor.h"

#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace hlir {
namespace framework {

void _Tensor_::set_type(Type type) {
  type_ = type;
  switch (type.type()) {
    case common::Type::type_t::Int:
      buffer_->data()->type = cinn_int32_t();
      break;
    case common::Type::type_t::Float:
      buffer_->data()->type = cinn_float32_t();
      break;
    default:
      buffer_->data()->type = cinn_unk_t();
      break;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

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

#include "infrt/common/dtype.h"

namespace infrt {

const char* DType::name() const {
  switch (kind_) {
#define CINNRT_DTYPE(enum__, value__) \
  case Kind::enum__:                  \
    return #enum__;                   \
    break;
#include "infrt/common/dtype.def"
#undef CINNRT_DTYPE
  }

  return "";
}

size_t DType::GetHostSize() const {
  switch (kind_) {
#define CINNRT_DTYPE(enum__, value__) \
  case DType::Kind::enum__:           \
    return sizeof(DTypeInternal<DType::Kind::enum__>::type);
#include "infrt/common/dtype.def"
#undef CINNRT_DTYPE
  }
  return 0;
}

}  // namespace infrt

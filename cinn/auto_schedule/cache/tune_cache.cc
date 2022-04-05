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

#include "cinn/auto_schedule/cache/tune_cache.h"

#include <cstring>

namespace cinn {
namespace auto_schedule {

TuneCache::TuneCache(size_t capacity) : cost_to_result_(capacity) {}

bool TuneCache::Save(const std::string& path) {
  // TODO(zhhsplendid): Implement this function
  return false;
}

bool TuneCache::Load(const std::string& load) {
  // TODO(zhhsplendid): Implement this function
  return false;
}

}  // namespace auto_schedule
}  // namespace cinn

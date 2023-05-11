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

#include <string>

#include "cinn/common/target.h"

namespace cinn {
namespace runtime {

using common::Target;

bool CheckStringFlagTrue(const std::string &flag);
bool CheckStringFlagFalse(const std::string &flag);

void SetCinnCudnnDeterministic(bool state);
bool GetCinnCudnnDeterministic();

class RandomSeed {
 public:
  static unsigned long long GetOrSet(unsigned long long seed = 0);
  static unsigned long long Clear();

 private:
  RandomSeed()                   = default;
  RandomSeed(const RandomSeed &) = delete;
  RandomSeed &operator=(const RandomSeed &) = delete;

  static unsigned long long seed_;
};

class CurrentTarget {
 public:
  static Target &GetCurrentTarget();
  static void SetCurrentTarget(const Target &target);

 private:
  CurrentTarget()                      = default;
  CurrentTarget(const CurrentTarget &) = delete;
  CurrentTarget &operator=(const CurrentTarget &) = delete;

  static bool IsCompiledWithCUDA() {
#ifdef CINN_WITH_CUDA
    return true;
#else
    return false;
#endif
  }

  static Target target_;
};

}  // namespace runtime
}  // namespace cinn

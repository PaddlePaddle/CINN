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

namespace cinn {
namespace runtime {

void SetCinnCudnnDeterministic(bool state);
bool GetCinnCudnnDeterministic();

class RandomSeed {
 public:
  static unsigned int GetOrSet(unsigned int seed = 0);
  static unsigned int Clear();

 private:
  RandomSeed()                   = default;
  RandomSeed(const RandomSeed &) = delete;
  RandomSeed &operator=(const RandomSeed &) = delete;

  static unsigned int seed_;
};

}  // namespace runtime
}  // namespace cinn

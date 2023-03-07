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

#pragma once
#include "cinn/frontend/net_builder.h"

namespace cinn {
namespace auto_schedule {

class TestProgramBuilder {
 public:
  virtual frontend::Program operator()() = 0;
};

class TestOpBuilder : public TestProgramBuilder {
 public:
  TestOpBuilder(const std::string& name) : builder_(name) {}
  frontend::Program operator()() { return builder_.Build(); }

 protected:
  frontend::NetBuilder builder_;
};

}  // namespace auto_schedule
}  // namespace cinn

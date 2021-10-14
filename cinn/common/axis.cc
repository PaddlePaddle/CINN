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

#include "cinn/common/axis.h"

#include "cinn/common/common.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace common {

std::vector<ir::Var> GenDefaultAxis(int naxis) {
  std::vector<ir::Var> axis;
  for (int i = 0; i < naxis; i++) {
    axis.emplace_back(common::axis_name(i));
    CHECK(axis.back()->type().valid());
  }
  return axis;
}

std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis) {
  auto vars = GenDefaultAxis(naxis);
  std::vector<Expr> res;
  for (auto& v : vars) {
    res.push_back(Expr(v));
  }
  return res;
}

static std::set<std::string> axis_set() {
  static std::set<std::string> x(kAxises.begin(), kAxises.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) { return axis_set().count(x); }

}  // namespace common
}  // namespace cinn

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
#include <glog/logging.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cinn {
namespace ir {

struct Var;
struct Expr;

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace common {

const static std::vector<std::string> kAxises({
    "i",  // level 0
    "j",  // level 1
    "k",  // level 2
    "a",  // level 3
    "b",  // level 4
    "c",  // level 5
    "d",  // level 6
    "e",  // level 7
    "f",  // level 8
    "g",  // level 9
    "h",  // level 10
    "l",  // level 11
    "m",  // level 12
    "n",  // level 13
    "o",  // level 14
    "p",  // level 15
    "q",  // level 16
    "r",  // level 17
    "s",  // level 18
    "t",  // level 19
    "u",  // level 20
    "v",  // level 21
    "w",  // level 22
    "x",  // level 23
    "y",  // level 24
    "z"   // level 25
});

//! Get the predifined axis name.
inline const std::string& axis_name(int level) {
  CHECK_LT(level, kAxises.size())
      << "The loop level should be less than 26. Please check if any variables have dimensions greater than 26.";
  return kAxises[level];
}

//! Generate `naxis` axis using the global names (i,j,k...).
std::vector<ir::Var> GenDefaultAxis(int naxis);
std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis);

bool IsAxisNameReserved(const std::string& x);

}  // namespace common
}  // namespace cinn

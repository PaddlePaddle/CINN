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
#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {

template <class T>
T GetAttr(const absl::flat_hash_map<std::string, framework::AttrType>& attrs, const std::string& key, const T&& value) {
  if (attrs.find(key) != attrs.end()) {
    return absl::get<T>(attrs.at(key));
  } else {
    return value;
  }
}

template <typename T = int>
std::vector<Expr> ToCinnExprs(const std::vector<T>& args) {
  std::vector<Expr> exprs;
  std::transform(args.begin(), args.end(), std::back_inserter(exprs), [](const T& arg) { return Expr(arg); });
  return exprs;
}

using CINNSchedule = lang::PackedFunc;

CINNSchedule GetElementwiseScheduleFunc(const std::vector<std::vector<int>>& output_shapes,
                                        const Target& target,
                                        bool vectorizable = true);

CINNSchedule GetInjectiveScheduleFunc(const std::vector<std::vector<int>>& output_shapes,
                                      const Target& target,
                                      bool vectorizable = true);

}  // namespace hlir
}  // namespace cinn

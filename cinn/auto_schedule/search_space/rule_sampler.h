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

#include <memory>
#include <random>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"

namespace cinn {
namespace auto_schedule {

class SearchState;

class rulesampler {
 public:
  // Create a rulesampler with the specific strategy name
  // and necessary construct parameters.
  static std::unique_ptr<rulesampler> Make(const std::vector<AutoGenRule*>& potential_rules,
                                           const std::string& strategy     = "traversal",
                                           const std::vector<int>& weights = {});
  // Return the name of schedule strategy
  virtual const char* Name() const = 0;

  // Reset associated states to schedule at the beginning
  virtual void Reset() = 0;

  // Select a rule to apply
  virtual AutoGenRule* NextRule(bool remove = true) = 0;

 protected:
  // A rulesampler object should be created with the static function Make()
  rulesampler(const std::vector<AutoGenRule*>& potential_rules) : potential_rules_(&potential_rules) {}

  // The pointer refers to all potential rules
  const std::vector<AutoGenRule*>* potential_rules_;
};

// Schedule rules with traversal strategy,
// witch means to select rules one by one until all rules are traversed.
class Traversalrulesampler : public rulesampler {
 public:
  Traversalrulesampler(const std::vector<AutoGenRule*>& potential_rules) : rulesampler(potential_rules), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

  AutoGenRule* NextRule(bool remove = true) override;

 private:
  int cur_idx_;
};

// Schedule rules with probabilistic strategy,
// witch means randomly picking rules according to the given distribution.
class Probabilisticrulesampler : public rulesampler {
 public:
  Probabilisticrulesampler(const std::vector<AutoGenRule*>& potential_rules, const std::vector<int>& weights = {});

  const char* Name() const override { return "probabilistic"; }

  void Reset() override {}

  AutoGenRule* NextRule(bool remove = true) override;

 private:
  std::vector<int> weights_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::discrete_distribution<> distribution_;
  int remains_;
};

}  // namespace auto_schedule
}  // namespace cinn

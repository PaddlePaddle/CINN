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

#include "cinn/auto_schedule/measure/measure.h"
#include "cinn/auto_schedule/search_space/search_state.h"

namespace cinn {
namespace auto_schedule {

// Record related data about tuning process of a measure candidate
struct TuningRecord {
  // the unique key to identify a task
  std::string task_key;
  // the cost time of the candidate executed during measure
  double execution_cost;  // unit: us
  // the searched candidate to be saved
  SearchState state;

  // a binary compare function that denotes when the left
  // will be sorted in the front of the right
  struct Compare {
    bool operator()(const TuningRecord& lhs, const TuningRecord& rhs) const;
  };
};

// A database supports insert or lookup historial tuning result with sepecified traits.
// It can be implemented with a concrete storage to save/load underlying data,
// such as memory, file, database server.
class Database {
 public:
  Database()  = default;
  ~Database() = default;
  // add a record into the database
  virtual bool AddRecord(TuningRecord&& record) = 0;
  // return all records whose task_keys are equal to the specified key
  virtual std::vector<TuningRecord> LookUp(const std::string& task_key) = 0;
  // return the states of the top k in sorted candidates
  virtual std::vector<SearchState> GetTopK(const std::string& task_key, int k) = 0;
};

}  // namespace auto_schedule
}  // namespace cinn

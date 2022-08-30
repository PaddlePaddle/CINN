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

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include "cinn/auto_schedule/auto_schedule.pb.h"
#include "cinn/auto_schedule/measure/measure.h"
#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/ir/ir_schedule.h"

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

  TuningRecord() = default;

  // initialize a TuningRecord object from a proto object
  TuningRecord(const proto::TuningRecord& record_proto)
      : task_key(record_proto.task_key()), execution_cost(record_proto.execution_cost()), state(ir::ModuleExpr()) {}

  TuningRecord(const std::string& task_key, double execution_cost, const SearchState& state)
      : task_key(task_key), execution_cost(execution_cost), state(state) {}

  // convert to proto object
  proto::TuningRecord ToProto() const;
};

// A database supports insert or lookup historial tuning result with sepecified traits.
// It can be implemented with a concrete storage to save/load underlying data,
// such as memory, file, database server and so on, this base class can be regarded as
// one using memory as its underlying storage medium.
class Database {
 public:
  explicit Database(int capacity_per_task);
  ~Database() = default;

  // add a record into the database
  bool AddRecord(TuningRecord&& record);
  // return all records whose task_keys are equal to the specified key
  std::vector<TuningRecord> LookUp(const std::string& task_key);
  // return the states of the top k in sorted candidates
  std::vector<SearchState> GetTopK(const std::string& task_key, int k);
  // return the total number of stored candidates
  size_t Size();
  // return the number of stored candidates with specified key
  size_t Count(const std::string& task_key);

 protected:
  // commit the newly added record into underlying storage
  virtual bool Commit(const TuningRecord& record) { return true; }

  // map task_key to its records
  std::unordered_map<std::string, std::multiset<TuningRecord, TuningRecord::Compare>> key2record_;
  // the max number of candidates stored
  const int capacity_per_task_;
};

}  // namespace auto_schedule
}  // namespace cinn

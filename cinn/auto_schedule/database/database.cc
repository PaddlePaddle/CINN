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

#include "cinn/auto_schedule/database/database.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

bool TuningRecord::Compare::operator()(const TuningRecord& lhs, const TuningRecord& rhs) const {
  return lhs.execution_cost < rhs.execution_cost;
}

proto::TuningRecord TuningRecord::ToProto() const {
  proto::TuningRecord record_proto;
  record_proto.set_task_key(task_key);
  record_proto.set_execution_cost(execution_cost);

  return record_proto;
}

std::string TuningRecord::ToJSON() const {
  auto record_proto = ToProto();

  std::string json_string;
  auto status = google::protobuf::util::MessageToJsonString(record_proto, &json_string);
  CHECK(status.ok()) << "Failed to serialize record to JSON, task key = " << task_key;
  VLOG(0) << "json_string = \n" << json_string;

  return json_string;
}

Database::Database(int capacity_per_task) : capacity_per_task_(capacity_per_task) {
  CHECK_GT(capacity_per_task_, 0) << "capacity_per_task_ should be greater than 0";
}

bool Database::AddRecord(TuningRecord&& record) {
  CHECK(!record.task_key.empty()) << "task_key of TuningRecord can't be empty";
  Commit(record);

  auto& records = key2record_[record.task_key];
  records.emplace(record);
  if (records.size() > capacity_per_task_) {
    records.erase(std::prev(records.end()));
  }
  return true;
}

std::vector<TuningRecord> Database::LookUp(const std::string& task_key) {
  auto fit = key2record_.find(task_key);
  if (fit == key2record_.end()) {
    return {};
  }

  std::vector<TuningRecord> results;
  results.reserve(fit->second.size());
  results.assign(fit->second.begin(), fit->second.end());
  return results;
}

std::vector<SearchState> Database::GetTopK(const std::string& task_key, int k) {
  auto fit = key2record_.find(task_key);
  if (fit == key2record_.end() || k <= 0) {
    return {};
  }
  if (k > capacity_per_task_) {
    LOG(WARNING) << "Input k:" << k << " is greater than the capacity";
    k = capacity_per_task_;
  }

  std::vector<SearchState> results;
  results.reserve(k);
  for (const TuningRecord& record : fit->second) {
    results.emplace_back(record.state);
    if (results.size() == k) {
      break;
    }
  }
  return results;
}

size_t Database::Size() {
  auto res =
      std::accumulate(key2record_.begin(), key2record_.end(), size_t(0), [](size_t res, const auto& kv) -> size_t {
        return std::move(res) + kv.second.size();
      });
  return res;
}

size_t Database::Count(const std::string& task_key) {
  auto fit = key2record_.find(task_key);
  if (fit == key2record_.end()) {
    return 0;
  }
  return fit->second.size();
}

}  // namespace auto_schedule
}  // namespace cinn

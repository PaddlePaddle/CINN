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

#include "cinn/auto_schedule/database/memory_database.h"

#include <vector>

namespace cinn {
namespace auto_schedule {

bool MemoryDatabase::AddRecord(TuningRecord&& record) {
  CHECK(!record.task_key.empty()) << "task_key of TuningRecord can't be empty";
  auto& records = key2record_[record.task_key];
  records.insert(record);
  return true;
}

std::vector<TuningRecord> MemoryDatabase::LookUp(const std::string& task_key) {
  auto fit = key2record_.find(task_key);
  if (fit == key2record_.end()) {
    return {};
  }

  std::vector<TuningRecord> results;
  results.reserve(it->second.size());
  results.assign(it->second.begin(), it->second.end());
  return results;
}

std::vector<SearchState> MemoryDatabase::GetTopK(const std::string& task_key, int k) {
  auto fit = key2record_.find(task_key);
  if (fit == key2record_.end() || k <= 0) {
    return {};
  }

  std::vector<SearchState> results;
  results.reserve(k);
  for (const TuningRecord& record : it->second) {
    results.emplace_back(record.state);
    if (results.size() == k) {
      break;
    }
  }
  return results;
}

}  // namespace auto_schedule
}  // namespace cinn

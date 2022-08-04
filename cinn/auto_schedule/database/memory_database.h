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

#include <set>
#include <unordered_map>

#include "cinn/auto_schedule/database/database.h"

namespace cinn {
namespace auto_schedule {

// A concrete database that save/load underlying data in memory
class MemoryDatabase : public Database {
 public:
  MemoryDatabase(int capacity_per_task);
  ~MemoryDatabase() = default;
  bool AddRecord(TuningRecord&& record) override;
  std::vector<TuningRecord> LookUp(const std::string& task_key) override;
  std::vector<SearchState> GetTopK(const std::string& task_key, int k) override;
  size_t Size() override;

 private:
  // map task_key to its records
  std::unordered_map<std::string, std::multiset<TuningRecord, TuningRecord::Compare>> key2record_;
  // the max number of candidates stored
  const int capacity_per_task_;
};

}  // namespace auto_schedule
}  // namespace cinn

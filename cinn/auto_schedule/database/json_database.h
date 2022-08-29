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

#include "database.h"

namespace cinn {
namespace auto_schedule {

// JSONDatabase is a database implemented by JSON file to save/load underlying data.
class JSONDatabase : public Database {
 public:
  /*!
   * \brief Build a JSONDatabase object from a json file.
   * \param capacity_per_task The max number of candidates stored.
   * \param tuning_record_file The name of the json file.
   * \param allow_new_file Whether to create new file when the given path is not found.
   */
  explicit JSONDatabase(int capacity_per_task, const std::string& tuning_record_file, bool allow_new_file);
  ~JSONDatabase() = default;

  /*!
   * \brief Reinitialize the JSONDatabase object from a json file.
   * \param tuning_record_file The name of the json file.
   * \param allow_new_file Whether to create new file when the given path is not found.
   */
  void ReInit(const std::string& tuning_record_file, bool allow_new_file);

 protected:
  // commit the newly added record into json file
  bool Commit(const TuningRecord& record) override;

  // the name of the json file to save tuning records.
  std::string tuning_record_file_;

  std::mutex mtx_;
};

}  // namespace auto_schedule
}  // namespace cinn

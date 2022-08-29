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

#include "cinn/auto_schedule/database/json_database.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include <fstream>

#include "cinn/auto_schedule/database/tuning_record.pb.h"
#include "cinn/utils/multi_threading.h"

namespace cinn {
namespace auto_schedule {

/*!
 * \brief Append a line to a json file.
 * \param file The name of the json file.
 * \param line The line to append.
 */
void JSONFileAppendLine(const std::string& file, const std::string& line) {
  std::ofstream os(file, std::ofstream::app);
  CHECK(os.good()) << "Cannot open the file to write: " << file;
  os << line << std::endl;
}

/*!
 * \brief Read lines from a json file.
 * \param file The name of the json file.
 * \param allow_new_file Whether to create new file when the given path is not found.
 * \return An array containing lines read from the json file.
 */
std::vector<std::string> JSONFileReadLines(const std::string& file, bool allow_new_file = true) {
  std::ifstream is(file);
  if (is.good()) {
    std::vector<std::string> json_strs;
    for (std::string str; std::getline(is, str);) {
      json_strs.push_back(str);
    }

    return json_strs;
  }
  CHECK(allow_new_file) << "File doesn't exist: " << file;
  std::ofstream os(file);
  CHECK(os.good()) << "Cannot create new file: " << file;
  return {};
}

// convert a TuningRecord object to string in JSON format
std::string RecordToJSON(const TuningRecord& record) {
  cinn::auto_schedule::proto::TuningRecord record_proto;
  record_proto.set_task_key(record.task_key);
  record_proto.set_execution_cost(record.execution_cost);

  std::string json_string;
  auto status = google::protobuf::util::MessageToJsonString(record_proto, &json_string);
  CHECK(status.ok()) << "Failed to serialize record to JSON, task key = " << record.task_key;
  VLOG(0) << "json_string = \n" << json_string;

  return json_string;
}

// convert a line of string in JSON format to a TuningRecord object
TuningRecord JSONToRecord(const std::string& json_string) {
  cinn::auto_schedule::proto::TuningRecord record_proto;
  auto status = google::protobuf::util::JsonStringToMessage(json_string, &record_proto);
  CHECK(status.ok()) << "Failed to parse JSON: " << json_string;

  return TuningRecord(
      {record_proto.task_key(), record_proto.execution_cost(), SearchState(std::move(ir::ModuleExpr()))});
}

JSONDatabase::JSONDatabase(int capacity_per_task, const std::string& tuning_record_file, bool allow_new_file)
    : Database(capacity_per_task), tuning_record_file_(tuning_record_file) {
  auto json_lines = JSONFileReadLines(tuning_record_file_, allow_new_file);

  // for (const auto& line : json_lines) {
  //   auto record = BuildTuningRecord(line);

  //   auto& records = this->key2record_[record->task_key];
  //   records.emplace(*record);
  //   if (records.size() > this->capacity_per_task_) {
  //     records.erase(std::prev(records.end()));
  //   }
  // }

  auto worker_fn = [this, &json_lines](int index) {
    auto record = BuildTuningRecord(json_lines[index]);

    std::lock_guard<std::mutex> lk(mtx_);
    auto& records = this->key2record_[record->task_key];
    records.emplace(*record);
    if (records.size() > this->capacity_per_task_) {
      records.erase(std::prev(records.end()));
    }
  };
  utils::parallel_run(worker_fn, utils::SequenceDispatcher(0, json_lines.size()), -1);
}

void JSONDatabase::ReInit(const std::string& tuning_record_file, bool allow_new_file) {
  tuning_record_file_ = tuning_record_file;
  key2record_.clear();
  auto json_lines = JSONFileReadLines(tuning_record_file_, allow_new_file);

  auto worker_fn = [this, &json_lines](int index) {
    auto record = BuildTuningRecord(json_lines[index]);

    std::lock_guard<std::mutex> lk(mtx_);
    auto& records = this->key2record_[record->task_key];
    records.emplace(*record);
    if (records.size() > this->capacity_per_task_) {
      records.erase(std::prev(records.end()));
    }
  };
  utils::parallel_run(worker_fn, utils::SequenceDispatcher(0, json_lines.size()), -1);
}

bool JSONDatabase::Commit(const TuningRecord& record) {
  std::string json_string = record.ToJSON();
  JSONFileAppendLine(tuning_record_file_, json_string);

  return true;
}

}  // namespace auto_schedule
}  // namespace cinn

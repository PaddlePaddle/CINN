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

#include <gflags/gflags.h>

#include <mutex>
#include <string>

#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {

namespace auto_schedule {

// Global task registrar, used to save the initial ModuleExpr of each task.
class TaskRegistry {
 public:
  static TaskRegistry* Global() {
    static TaskRegistry task_registry;
    return &task_registry;
  }

  // Store the initial ModuleExpr of a task into the map
  void Regist(const std::string& task_key, const ir::ModuleExpr& module_expr) {
    std::lock_guard<std::mutex> lock(mtx_);
    task_map_.insert({task_key, optim::IRCopy(module_expr)});
  }

  // Get the initial ModuleExpr of a task by serialized_key;
  const ir::ModuleExpr& Get(const std::string& task_key) {
    std::lock_guard<std::mutex> lock(mtx_);
    CHECK(task_map_.find(task_key) != task_map_.end()) << "task with task_key = " << task_key << "is not exist.";
    return task_map_.at(task_key);
  }

  // Get the initial ModuleExpr of a task by serialized_key;
  bool Remove(const std::string& task_key) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (task_map_.find(task_key) == task_map_.end()) {
      return false;
    }
    task_map_.erase(task_key);
    return true;
  }

  // Check if the task_key exists;
  bool Has(const std::string& task_key) {
    std::lock_guard<std::mutex> lock(mtx_);
    return task_map_.count(task_key) != 0;
  }

 private:
  TaskRegistry()                    = default;
  TaskRegistry(const TaskRegistry&) = delete;
  void operator=(TaskRegistry&) = delete;

  std::mutex mtx_;
  absl::flat_hash_map<std::string, ir::ModuleExpr> task_map_;
};

}  // namespace auto_schedule
}  // namespace cinn

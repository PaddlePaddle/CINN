// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cinn {
namespace utils {

enum class EventType {
  // kOrdinary is default type
  kOrdinary,
  // kGraph is frontend Graph process
  kGraph,
  // kProgram is fronted Program process
  kProgram,
  // kPass is Graph and Program pass process
  kPass,
  // kOpLowering is NetBuilder OpLower process
  kOpLowering,
  // kSchedule is applying Schedule process
  kSchedule,
  // kCodeGen is AstCodegen process
  kCodeGen,
  // kCompile is LLVM or CUDA NVTX compile process
  kCompile,
  // kInstruction is running instruction process
  kInstruction
};

inline std::string EventTypeToString(const EventType& type);

std::ostream& operator<<(std::ostream& os, const EventType& type);

struct HostEvent {
  std::string annotation_;
  double duration_;  // ms
  EventType type_;

  HostEvent(const std::string& annotation, double duration, EventType type)
      : annotation_(annotation), duration_(duration), type_(type) {}
};

class Summary {
 public:
  struct Raito {
    double value;
    Raito(double val) : value(val){};
    std::string ToStr() const { return std::to_string(value); }
  };

  struct Item {
    const HostEvent* event;
    Raito sub_raito{0.0};    // percentage of EventType
    Raito total_raito{0.0};  // precentage of total process

    Item(const HostEvent& e) : event(&e) {}
    bool operator<(const Item& other) const { return total_raito.value > other.total_raito.value; }
  };

  static std::string Format(const std::vector<HostEvent>& events);

  static std::string AsStr(const std::vector<Item>& items);
};

class HostEventRecorder {
 public:
  // singleton
  static HostEventRecorder& GetInstance() {
    static HostEventRecorder instance;
    return instance;
  }

  static std::string Table() { return Summary::Format(GetInstance().Events()); }

  void Clear() { events_.clear(); }

  std::vector<HostEvent>& Events() { return events_; }

  void RecordEvent(const std::string& annotation, double duration, EventType type) {
    GetInstance().Events().emplace_back(annotation, duration, type);
  }

 private:
  std::vector<HostEvent> events_;
};

}  // namespace utils
}  // namespace cinn

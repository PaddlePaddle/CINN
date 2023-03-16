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

#include "cinn/utils/event.h"

namespace cinn {
namespace utils {
inline std::string EventTypeToString(const EventType &type) {
  switch (type) {
    case EventType::kOrdinary:
      return "Ordinary";
    case EventType::kGraph:
      return "Graph";
    case EventType::kProgram:
      return "Program";
    case EventType::kPass:
      return "Pass";
    case EventType::kOpLowering:
      return "OpLowering";
    case EventType::kSchedule:
      return "Schedule";
    case EventType::kCodeGen:
      return "CodeGen";
    case EventType::kCompile:
      return "Compile";
    case EventType::kInstruction:
      return "Instruction";
    default:
      return "";
  }
}

std::ostream &operator<<(std::ostream &os, const EventType &type) {
  os << EventTypeToString(type).c_str();
  return os;
}

std::string Summary::Format(const std::vector<HostEvent> &events) {
  std::vector<Item> items;
  std::unordered_map<EventType, double> category_cost;

  double total_cost = 0.0;
  for (auto &e : events) {
    items.emplace_back(e);
    category_cost[e.type_] += e.duration_;
    total_cost += e.duration_;
  }
  // Calculate Ratio
  for (auto &item : items) {
    item.sub_raito   = item.event->duration_ / category_cost[item.event->type_] * 100.0;
    item.total_raito = item.event->duration_ / total_cost * 100.0;
  }

  std::sort(items.begin(), items.end());

  return AsStr(items);
}

std::string Summary::AsStr(const std::vector<Item> &items) {
  std::ostringstream os;
  os << "\n";
  os << "Category\t\t"
     << "Name\t\t"
     << "CostTime(ms)\t\t"
     << "Ratio in Category(%)\t\t"
     << "Ratio in Total(%)\n";

  for (auto &item : items) {
    os << item.event->type_ << "\t" << item.event->annotation_.c_str() << "\t" << item.event->duration_ << "\t"
       << item.sub_raito.value << "\t" << item.total_raito.value;
    os << "\n";
  }
  return os.str();
}

}  // namespace utils
}  // namespace cinn

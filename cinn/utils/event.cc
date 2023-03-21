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

#include <glog/logging.h>  // for GLog

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
    case EventType::kFusePass:
      return "FusePass";
    case EventType::kCompute:
      return "Compute";
    case EventType::kSchedule:
      return "Schedule";
    case EventType::kOptimize:
      return "Optimize";
    case EventType::kCodeGen:
      return "CodeGen";
    case EventType::kCompile:
      return "Compile";
    case EventType::kInstruction:
      return "Instruction";
    default:
      LOG(FATAL) << "Unknown event type";
  }
}

std::ostream &operator<<(std::ostream &os, const EventType &type) {
  os << EventTypeToString(type).c_str();
  return os;
}

std::string Summary::Format(const std::vector<HostEvent> &events) {
  std::vector<Item> items;
  std::unordered_map<EventType, double> category_cost;

  double total_cost     = 0.0;
  size_t max_annot_size = 20;
  for (auto &e : events) {
    items.emplace_back(e);
    category_cost[e.type_] += e.duration_;
    total_cost += e.duration_;
    max_annot_size = std::max(max_annot_size, e.annotation_.size());
  }
  // Calculate Ratio
  for (auto &item : items) {
    item.sub_raito   = item.event->duration_ / category_cost[item.event->type_] * 100.0;
    item.total_raito = item.event->duration_ / total_cost * 100.0;
  }

  std::sort(items.begin(), items.end());

  return AsStr(items, /*data_width=*/max_annot_size);
}

std::string Summary::AsStr(const std::vector<Item> &items, int data_width) {
  std::ostringstream os;

  os << "\n\n------------------------->     Profiling Report     <-------------------------\n\n";

  std::vector<std::string> titles = {"Category", "Name", "CostTime(ms)", "Ratio in Category(%)", "Ratio in Total(%)"};

  size_t pad_size = 0;
  for (auto &t : titles) {
    pad_size = data_width > t.size() ? data_width - t.size() : 1;
    os << ' ' << t << std::string(pad_size, ' ');
  }

  os << "\n\n";

  for (auto &item : items) {
    std::vector<std::string> infos = {EventTypeToString(item.event->type_),
                                      item.event->annotation_,
                                      std::to_string(item.event->duration_),
                                      item.sub_raito.ToStr(),
                                      item.total_raito.ToStr()};
    for (auto &info : infos) {
      pad_size = data_width > info.size() ? data_width - info.size() : 1;
      os << ' ' << info << std::string(pad_size, ' ');
    }
    os << "\n";
  }
  os << "\n";
  return os.str();
}

}  // namespace utils
}  // namespace cinn

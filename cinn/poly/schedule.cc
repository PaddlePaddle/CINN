#include "cinn/poly/schedule.h"
#include "cinn/utils/string.h"

#include <sstream>

namespace cinn {
namespace poly {

std::string TimeSchedule::__str__() const {
  CHECK(!time_dims.empty());

  // generate range: [dup, t0, t1...]
  std::vector<std::string> range_dims({"dup"});
  for (int i = 0; i < time_dims.size(); i++) {
    range_dims.push_back("t" + std::to_string(i));
  }

  // generate conditions
  std::vector<std::string> conds;
  for (int i = 0; i < time_dims.size(); i++) {
    conds.push_back(std::to_string(time_dims[i].time));
    conds.push_back(time_dims[i].dim);
  }

  return utils::StringFormat("{ %s[%s] -> [%s]: %s",
                             id.c_str(),
                             utils::Join(domain_dims, ", ").c_str(),
                             utils::Join(range_dims, ", ").c_str(),
                             utils::Join(conds, " and ").c_str());
}

void Scheduler::RegisterElement(const Element &x) {
  CHECK(!registration_finalized_) << "element registration has been finalized.";
  space_size_ = std::max(space_size_, isl_map_dim(x.schedule().get(), isl_dim_out));

  // Use the dimensions from element's schedule's range as the new domain dimensions because in Element, the schedule is
  // like '{ S0[i,j] -> S0[i_outer, i_inner, j] }', the scheduler should schedule base on the range.
  TimeSchedule schedule(GetDimNames(x.schedule(), isl_dim_out));
  schedule_.emplace(x.id(), std::move(schedule));
}

void Scheduler::FinalizeRegistration() {
  CHECK_GT(space_size_, 0) << "No valid dimension is collected, use RegisterElement to collect some elements";
  CHECK(!schedule_.empty()) << "No valid dimension is collected, use RegisterElement to collect some elements";
  registration_finalized_ = false;

  for (auto &item : schedule_) {
    item.second.ResizeTimeSpace(space_size_);
  }
}

Scheduler &Scheduler::After(const Element &a, const Element &b, int level) {
  CHECK_LT(level, space_size_);
  depend_flow_graph_[b.id()].depend_level[a.id()] = level;
  return *this;
}

Scheduler &Scheduler::Before(const Element &a, const Element &b, int level) { return After(b, a, level); }

std::unordered_map<std::string, isl::map> Scheduler::BuildSchedule() const {}

}  // namespace poly
}  // namespace cinn

#include "cinn/poly/schedule.h"
#include <sstream>
#include "cinn/common/graph_utils.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

/**
 * Node in the schedule graph.
 */
struct ScheduleGraphNode : public common::GraphNode {
  TimeSchedule time_schedule;

  explicit ScheduleGraphNode(const std::vector<std::string> &dims) : time_schedule(dims) {}
};

struct ScheduleGraphEdge : public common::GraphEdge {
  ScheduleGraphEdge(common::GraphNode *a, common::GraphNode *b) : common::GraphEdge(a, b) {}

  //! Dependency level.
  int level;
};

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
  schedule_graph_.RegisterNode(x.id(), common::make_shared<ScheduleGraphNode>(GetDimNames(x.schedule(), isl_dim_out)));
}

void Scheduler::FinalizeRegistration() {
  CHECK_GT(space_size_, 0) << "No valid dimension is collected, use RegisterElement to collect some elements";
  CHECK(!schedule_graph_.nodes().empty())
      << "No node is registered to the graph, use RegisterElement to collect some elements";
  registration_finalized_ = true;

  for (auto &item : schedule_graph_.nodes()) {
    item->As<ScheduleGraphNode>()->time_schedule.ResizeTimeSpace(space_size_);
  }
}

Scheduler &Scheduler::After(const Element &a, const Element &b, int level) {
  CHECK_LT(level, space_size_);
  auto *a_node = schedule_graph_.RetriveNode(a.id())->As<ScheduleGraphNode>();
  auto *b_node = schedule_graph_.RetriveNode(a.id())->As<ScheduleGraphNode>();
  CHECK(a_node) << "no node called " << a.id() << " registered in the graph";
  CHECK(b_node) << "no node called " << b.id() << " registered in the graph";

  common::GraphEdge *a_edge, *b_edge;
  std::tie(a_edge, b_edge) = a_node->LinkTo<ScheduleGraphEdge>(b_node);
  a_edge->As<ScheduleGraphEdge>()->level = level;
  b_edge->As<ScheduleGraphEdge>()->level = level;
  return *this;
}

Scheduler &Scheduler::Before(const Element &a, const Element &b, int level) { return After(b, a, level); }

std::unordered_map<std::string, isl::map> Scheduler::BuildSchedule() const {}

}  // namespace poly
}  // namespace cinn

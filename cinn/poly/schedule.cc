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

  explicit ScheduleGraphNode(const std::string &id, const std::vector<std::string> &dims) : time_schedule(id, dims) {}
};

struct ScheduleGraphEdge : public common::GraphEdge {
  ScheduleGraphEdge(common::GraphNode *a, common::GraphNode *b) : common::GraphEdge(a, b) {}

  //! Dependency level.
  int level;
};

std::string TimeSchedule::__str__() const {
  CHECK(!time_dims.empty());

  // generate range: [dup, t0, t1...]
  std::vector<std::string> range_dims;
  for (int i = 0; i < time_dims.size(); i++) {
    range_dims.push_back("t" + std::to_string(i));
    range_dims.push_back("d" + std::to_string(i));
  }

  // generate conditions
  std::vector<std::string> conds;
  for (int i = 0; i < time_dims.size(); i++) {
    conds.push_back(utils::StringFormat("%s=%s", range_dims[2 * i].c_str(), std::to_string(time_dims[i].time).c_str()));
    conds.push_back(utils::StringFormat("%s=%s", range_dims[2 * i + 1].c_str(), time_dims[i].dim.c_str()));
  }

  return utils::StringFormat("{ %s[%s] -> [%s]: %s }",
                             id_.c_str(),
                             utils::Join(domain_dims, ", ").c_str(),
                             utils::Join(range_dims, ", ").c_str(),
                             utils::Join(conds, " and ").c_str());
}

TimeSchedule::TimeSchedule(const std::string &id, const std::vector<std::string> &dims) {
  id_         = id;
  domain_dims = dims;
  for (auto &dim : domain_dims) {
    time_dims.emplace_back(dim, 0);
  }
}

void TimeSchedule::OrderAfter(const TimeSchedule &other, int level) {
  CHECK_EQ(space_size(), other.space_size()) << "space not match";
  CHECK_LT(level, other.space_size());
  CHECK(!time_dims.empty());

  for (int i = 0; i <= level; i++) {
    this->time_dims[i].time = std::max(other.time_dims[i].time, this->time_dims[i].time);
  }

  this->time_dims[level].time++;
}

isl::map TimeSchedule::to_isl(isl::ctx ctx) const {
  VLOG(3) << "isl: " << __str__();
  return isl::map(ctx, __str__());
}

const std::string &TimeSchedule::id() const {
  CHECK(!id_.empty());
  return id_;
}

void Scheduler::RegisterElement(const Element &x) {
  CHECK(!registration_finalized_) << "element registration has been finalized.";
  space_size_ = std::max(space_size_, isl_map_dim(x.schedule().get(), isl_dim_out));
  VLOG(3) << "space_size: " << space_size_;

  // Use the dimensions from element's schedule's range as the new domain dimensions because in Element, the schedule is
  // like '{ S0[i,j] -> S0[i_outer, i_inner, j] }', the scheduler should schedule base on the range.
  auto dims      = GetDimNames(x.schedule(), isl_dim_out);
  std::string id = isl_map_get_tuple_name(x.schedule().get(), isl_dim_in);
  schedule_graph_.RegisterNode(x.id(),
                               common::make_shared<ScheduleGraphNode>(id, GetDimNames(x.schedule(), isl_dim_out)));

  if (!ctx_.get()) {
    ctx_ = x.domain().ctx();
  } else {
    CHECK_EQ(ctx_.get(), x.domain().ctx().get()) << "isl ctx not match";
  }
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
  auto *b_node = schedule_graph_.RetriveNode(b.id())->As<ScheduleGraphNode>();
  CHECK(a_node) << "no node called " << a.id() << " registered in the graph";
  CHECK(b_node) << "no node called " << b.id() << " registered in the graph";

  common::GraphEdge *a_edge, *b_edge;
  std::tie(a_edge, b_edge) = a_node->LinkTo<ScheduleGraphEdge>(b_node);
  a_edge->As<ScheduleGraphEdge>()->level = level;
  b_edge->As<ScheduleGraphEdge>()->level = level;
  return *this;
}

Scheduler &Scheduler::Before(const Element &a, const Element &b, int level) { return After(b, a, level); }

std::map<std::string, isl::map> Scheduler::BuildSchedule() const {
  std::map<std::string, isl::map> res;
  CHECK(ctx_.get());

  ScheduleGraph::node_order_t node_order;
  ScheduleGraph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = schedule_graph_.topological_order();
  for (auto *edge : edge_order) {
    auto *schedule_edge = edge->As<ScheduleGraphEdge>();
    auto *a_node        = schedule_graph_.RetriveNode(edge->source()->As<ScheduleGraphNode>()->time_schedule.id())
                       ->As<ScheduleGraphNode>();
    auto *b_node =
        schedule_graph_.RetriveNode(edge->sink()->As<ScheduleGraphNode>()->time_schedule.id())->As<ScheduleGraphNode>();
    CHECK(a_node);
    CHECK(b_node);

    int level = schedule_edge->level;
    b_node->time_schedule.OrderAfter(a_node->time_schedule, level);
  }

  for (auto *node : schedule_graph_.nodes()) {
    auto *schedule_node                    = node->As<ScheduleGraphNode>();
    res[schedule_node->time_schedule.id()] = schedule_node->time_schedule.to_isl(ctx_);
  }
  return res;
}

}  // namespace poly
}  // namespace cinn

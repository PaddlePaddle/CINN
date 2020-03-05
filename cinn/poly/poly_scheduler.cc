#include "cinn/poly/poly_scheduler.h"

namespace cinn {
namespace poly {

PolyScheduler &PolyScheduler::After(const Stage &a, const Stage &b, int level) {
  CHECK_LT(level, space_size_);
  auto *a_node = schedule_graph_.RetriveNode(a.id())->As<ScheduleGraphNode>();
  auto *b_node = schedule_graph_.RetriveNode(b.id())->As<ScheduleGraphNode>();
  CHECK(a_node) << "no node called " << a.id() << " registered in the graph";
  CHECK(b_node) << "no node called " << b.id() << " registered in the graph";

  common::GraphEdge *a_edge, *b_edge;
  std::tie(a_edge, b_edge)               = a_node->LinkTo<ScheduleGraphEdge>(b_node);
  a_edge->as<ScheduleGraphEdge>()->level = level;
  b_edge->as<ScheduleGraphEdge>()->level = level;
  return *this;
}

PolyScheduler &PolyScheduler::Before(const Stage &a, const Stage &b, int level) { return After(b, a, level); }

std::map<std::string, isl::map> PolyScheduler::BuildSchedule() const {
  std::map<std::string, isl::map> res;
  CHECK(ctx_.get());

  ScheduleGraph::node_order_t node_order;
  ScheduleGraph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = schedule_graph_.topological_order();
  for (auto *edge : edge_order) {
    auto *schedule_edge = edge->as<ScheduleGraphEdge>();
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

std::vector<std::string> PolyScheduler::WrapIteratorNames(const std::vector<std::string> &names) const {
  CHECK_EQ(names.size(), space_size());
  std::vector<std::string> res;
  for (int i = 0; i < space_size(); i++) {
    res.push_back("");        // fake name for time space.
    res.push_back(names[i]);  // name for the corresponding iterator.
  }
  return res;
}

PolyScheduler::PolyScheduler(const std::vector<Stage *> &stages) {
  CHECK_GT(stages.size(), 0) << "No stage is provided";
  for (auto *stage : stages) {
    AddStage(*stage);
  }
  FinishStageAdd();
}

}  // namespace poly
}  // namespace cinn

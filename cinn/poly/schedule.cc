#include "cinn/poly/schedule.h"

#include <deque>
#include <set>
#include <sstream>

#include "cinn/common/graph_utils.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/lang/tensor.h"
#include "cinn/poly/poly_scheduler.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

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

std::vector<std::string> TimeSchedule::final_axis_names() const {
  std::vector<std::string> dims;
  for (int i = 0; i < time_dims.size(); i++) {
    dims.push_back(std::to_string(time_dims[i].time).c_str());
    dims.push_back(time_dims[i].dim.c_str());
  }
  return dims;
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

void Schedule::PartitionGroups() {
  CHECK(!graph_->nodes().empty());
  groups_ = detail::PartitionGraphByIterationDomain(graph_);
}

void Schedule::ScheduleGroup(detail::Group *group) {
  CHECK(group);
  std::set<Stage *> dic;

  // create scheduler for this group.
  std::vector<Stage *> stages;
  for (auto &node : group->nodes) {
    dic.insert(node->stage.get());
    stages.push_back(node->stage.get());
  }
  PolyScheduler scheduler(stages);

  // NOTE this is unnecessary
  for (auto &node : group->nodes) {
    // if any outlink in the dic, schedule the output node After this by the last dimension.
    for (auto &outlink : node->outlinks()) {
      auto *out_node = outlink->sink()->As<DataFlowGraphNode>();
      if (dic.count(out_node->stage.get())) {
        int node_iter_dims = isl_set_dim(node->stage->transformed_domain().get(), isl_dim_set);
        int out_iter_dims  = isl_set_dim(out_node->stage->transformed_domain().get(), isl_dim_set);
        int level          = std::min(node_iter_dims, out_iter_dims) - 1;
        CHECK_GE(level, 0);
        scheduler.After(*node->stage, *out_node->stage, level);
      }
    }
  }
}

void Schedule::ScheduleEachGroup() {
  CHECK(!groups_.empty()) << "call PartitionGroups first";
  for (auto &group : groups_) {
    ScheduleGroup(&group);
  }
}

std::unique_ptr<Schedule> CreateSchedule(const ir::Tensor &tensor) {
  auto stages = GatherStagesInTensors({tensor});
  VLOG(3) << "collected " << stages.size() << " stages";
  return CreateSchedule(stages);
}

std::unique_ptr<Schedule> CreateSchedule(const std::vector<Stage *> &stages) {
  CHECK(!stages.empty());
  for (auto &stage : stages) {
    VLOG(4) << "stage: " << stage->domain();
  }
  auto graph = CreateGraph(stages);

  return std::unique_ptr<Schedule>(new Schedule(graph.get()));
}

std::vector<Stage *> GatherStagesInTensors(const std::vector<ir::Tensor> &xs, bool with_placeholder) {
  // get the stages from a tensor.
  std::vector<Stage *> stages;
  std::deque<ir::Tensor> queue;
  for (auto &x : xs) {
    CHECK(!x->inlined()) << "Inlined tensor should not be output of a function";
    queue.push_back(x);
  }

  std::set<Expr> visited;
  while (!queue.empty()) {
    auto top = queue.front();
    queue.pop_front();
    if (visited.count(Expr(top))) continue;
    visited.insert(Expr(top));
    if (top->stage()) {
      VLOG(3) << "collect stage " << top->stage();
      stages.push_back(top->stage());
    }

    auto tensor_exprs = ir::CollectIRNodes(Expr(top), [](const Expr *expr) { return expr->As<ir::_Tensor_>(); });
    for (auto &expr : tensor_exprs) {
      if (!visited.count(expr)) queue.push_back(ir::Tensor(const_cast<ir::_Tensor_ *>(expr.As<ir::_Tensor_>())));
    }
  }

  std::reverse(stages.begin(), stages.end());
  return stages;
}

void SchedulerBase::AddStage(const Stage &x) {
  CHECK(!registration_finalized_) << "element registration has been finalized.";
  space_size_ = std::max(space_size_, isl_map_dim(x.transform().get(), isl_dim_out));
  VLOG(3) << "space_size: " << space_size_;
  VLOG(3) << "schedule: " << x.transform();

  // Use the dimensions from element's schedule's range as the new domain dimensions because in Element, the schedule is
  // like '{ S0[i,j] -> S0[i_outer, i_inner, j] }', the scheduler should schedule base on the range.
  auto dims      = GetDimNames(x.transform(), isl_dim_out);
  std::string id = isl_map_get_tuple_name(x.transform().get(), isl_dim_in);
  schedule_graph_.RegisterNode(x.id(),
                               common::make_shared<ScheduleGraphNode>(id, GetDimNames(x.transform(), isl_dim_out)));
  // record the longest dimensions.
  if (dims.size() > detailed_dimension_names_.size()) detailed_dimension_names_ = dims;

  if (!ctx_.get()) {
    ctx_ = x.domain().ctx();
  } else {
    CHECK_EQ(ctx_.get(), x.domain().ctx().get()) << "isl ctx not match";
  }
}

void SchedulerBase::FinishStageAdd() {
  CHECK_GT(space_size_, 0) << "No valid dimension is collected, use RegisterElement to collect some elements";
  CHECK(!schedule_graph_.nodes().empty())
      << "No node is registered to the graph, use RegisterElement to collect some elements";
  registration_finalized_ = true;

  for (auto &item : schedule_graph_.nodes()) {
    VLOG(2) << "original dims in time_schedule: "
            << utils::Join(item->As<ScheduleGraphNode>()->time_schedule.domain_dims, ", ");
    item->As<ScheduleGraphNode>()->time_schedule.ResizeTimeSpace(space_size_);
  }
}

}  // namespace poly
}  // namespace cinn

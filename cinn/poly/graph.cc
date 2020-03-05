#include "cinn/poly/graph.h"

#include <deque>
#include <map>
#include <set>

namespace cinn {
namespace poly {

const DataFlowGraphNode* DataFlowGraphNode::group_ancestor() const {
  auto* p = this;
  while (p->group_parent) p = p->group_parent;
  return p;
}

DataFlowGraphNode* DataFlowGraphNode::group_ancestor() {
  auto* p = this;
  while (p->group_parent) p = p->group_parent;
  return p;
}

bool DataFlowGraphNode::TransformedDomainIsSame(const DataFlowGraphNode* a, const DataFlowGraphNode* b) {
  VLOG(3) << "a.domain " << a->stage->domain();
  VLOG(3) << "a.transform " << a->stage->transform();
  VLOG(3) << "b.domain " << b->stage->domain();
  VLOG(3) << "b.transform " << b->stage->transform();
  auto a_domain = a->stage->transformed_domain();
  auto b_domain = b->stage->transformed_domain();
  a_domain      = isl::manage(isl_set_set_tuple_name(a_domain.release(), ""));
  b_domain      = isl::manage(isl_set_set_tuple_name(b_domain.release(), ""));
  return isl_set_is_equal(a_domain.get(), b_domain.get());
}

int DataFlowGraphNode::group_height() const {
  int h   = 0;
  auto* p = this;
  while (p) {
    ++h;
    p = p->group_parent;
  }
  return h;
}

DataFlowGraphNode* DataFlowGraphNode::MergeGroup(DataFlowGraphNode* a, DataFlowGraphNode* b) {
  int ah      = a->group_height();
  int bh      = b->group_height();
  auto* a_anc = a->group_ancestor();
  auto* b_anc = b->group_ancestor();
  DataFlowGraphNode* common_anc{};
  if (ah < bh) {  // take a's ancestor
    b_anc->group_parent = a_anc;
    b->group_parent     = a_anc;
    return a_anc;
  } else {
    a_anc->group_parent = b_anc;
    a->group_parent     = b_anc;
    return b_anc;
  }
}
std::string DataFlowGraphNode::id() const {
  // NOTE the stage's id should be unique.
  return stage->id();
}

std::unique_ptr<DataFlowGraph> CreateGraph(const std::vector<Stage*>& stages) {
  std::map<std::string, Shared<DataFlowGraphNode>> id2stage;
  for (auto* x : stages) id2stage[x->id()] = make_shared<DataFlowGraphNode>(x);

  for (auto* stage : stages) {
    auto depend_statement_names = stage->input_statements();
    VLOG(3) << stage->id() << " depend " << utils::Join(depend_statement_names, ", ");
    for (auto& depend_statement : depend_statement_names) {
      auto input_it = id2stage.find(depend_statement);
      // We removed some node in the original stages(such as placeholders), so that there might be missing of some input
      // nodes, just ignore the dependence.
      if (input_it != std::end(id2stage)) {
        auto& input_node = id2stage.at(depend_statement);
        input_node->LinkTo(id2stage.at(stage->id()).get());
      }
    }
  }

  std::unique_ptr<DataFlowGraph> graph(new DataFlowGraph);
  for (auto& item : id2stage) graph->RegisterNode(item.first, item.second.get());
  VLOG(3) << "created graph:\n" << graph->Visualize();
  return graph;
}

}  // namespace poly
}  // namespace cinn

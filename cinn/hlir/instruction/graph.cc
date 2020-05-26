#include "cinn/hlir/instruction/graph.h"

namespace cinn {
namespace hlir {
namespace instruction {

std::unique_ptr<common::Graph> CreateComputationGraph(Computation* comp) {
  std::unique_ptr<common::Graph> graph(new common::Graph);
  for (const auto& instr : comp->instructions()) {
    LOG(INFO) << "register insturction: " << instr->to_debug_string();
    auto* cur_node = graph->RegisterNode(instr->id(), new InstructionGraphNode(instr.get()));
    for (int i = 0; i < instr->operand_count(); i++) {
      auto* arg      = instr->operand(i);
      auto* arg_node = graph->RetriveNode(arg->id());
      arg_node->LinkTo(cur_node);
    }
  }
  return graph;
}

ModuleGraph::ModuleGraph(Module* m) { Create(m); }

void ModuleGraph::Create(Module* m) {
  for (auto& item : m->computations()) {
    auto* comp        = item.second.get();
    auto graph        = CreateComputationGraph(comp);
    comp_graphs[comp] = std::move(graph);
  }
}

const char* InstructionGraphNode::type_info() const { return __type_info__; }

const char* InstructionGraphNode::__type_info__ = "InstructionGraphNode";
std::string InstructionGraphNode::id() const { return instruction->id(); }

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

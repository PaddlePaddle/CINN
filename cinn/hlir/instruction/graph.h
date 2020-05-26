#pragma once

#include <map>
#include <string>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/module.h"

namespace cinn {
namespace hlir {
namespace instruction {

struct InstructionGraphNode : public common::GraphNode {
  Instruction* instruction{};

  InstructionGraphNode(Instruction* x) : instruction(x) {}

  std::string id() const override;

  const char* type_info() const override;
  static const char* __type_info__;
};

struct ModuleGraph {
  std::map<Computation*, std::unique_ptr<common::Graph>> comp_graphs;

  ModuleGraph(Module* m);

 private:
  void Create(Module* m);
};

std::unique_ptr<common::Graph> CreateComputationGraph(Computation* comp);

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

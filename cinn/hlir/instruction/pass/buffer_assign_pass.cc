#include "cinn/hlir/instruction/pass/buffer_assign_pass.h"

#include "cinn/hlir/instruction/graph.h"
#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/pass.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace pass {

std::string_view BufferAssignPass::name() const { return name_; }

bool BufferAssignPass::Run(Module* module) {
  for (auto& item : module->computations()) {
    RunOnComputation(item.second.get());
  }
  return true;
}

bool BufferAssignPass::RunOnModuleGroup(ModuleGroup* module_group) { return false; }

bool BufferAssignPass::is_pass_pipeline() const { return PassInterface::is_pass_pipeline(); }

void BufferAssignPass::RunOnComputation(Computation* comp) {
  auto graph = CreateComputationGraph(comp);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  auto start_points = graph->start_points();
  // check all the start points are parameters
  for (auto& p : start_points) {
    auto* instruction = p->safe_as<InstructionGraphNode>()->instruction;
    CHECK(instruction);
    CHECK_EQ(instruction->instr_code(), InstrCode::Parameter) << instruction->to_debug_string();
  }

  // find the instruction those are able to inline.
  auto [topo_node_order, topo_edge_order] = graph->topological_order();
  for (auto& p : topo_node_order) {
    auto* instr_node = p->safe_as<InstructionGraphNode>();
    bool inlined     = ShouldInline(instr_node->instruction, comp);
    LOG(INFO) << instr_node->instruction->to_debug_string() << ", inlined: " << inlined;
    instr_node->instruction->set_inlined(inlined);
  }
}

bool BufferAssignPass::ShouldInline(Instruction* instr, Computation* comp) const {
  /*
   *    Unary -> Unary -> Unary -> Binary
   *       ^       ^        ^
   *    inline  inline   inline  notinline
   */
  bool is_output = instr == comp->output();
  if (is_output) return false;
  if (unary_codes_.count(instr->instr_code())) {
    return true;
  }

  if (instr->instr_code() == InstrCode::Call) return instr->As<CallInstruction>()->can_inlined();

  std::unordered_set<InstrCode> logical_ops({
      InstrCode::Tuple,
      InstrCode::TupleGet,
      InstrCode::Broadcast,
      InstrCode::Dot,
      InstrCode::Constant,
      InstrCode::Parameter,
      InstrCode::CustomCall,
      InstrCode::Input,
      InstrCode::Reduce,
      InstrCode::Transpose,
  });

  return !logical_ops.count(instr->instr_code());
}

}  // namespace pass
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

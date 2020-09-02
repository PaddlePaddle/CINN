#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {

void Instruction::PrepareOutputs() {
  auto* op_def = hlir::framework::OpRegistry::Global()->Find(get()->op_type);
  CHECK(op_def) << "No operator called [" << get()->op_type << "]";
  for (int i = 0; i < op_def->num_outputs; i++) {
    get()->outputs.push_back(Variable());
  }
}

Instruction::Instruction(std::string_view op_type, Program* parent)
    : common::Shared<_Instruction_>(common::make_shared<_Instruction_>()) {
  get()->op_type        = op_type;
  get()->parent_program = parent;
  PrepareOutputs();
}

Placeholder::operator Variable() {
  Variable var(id());
  var->shape = shape();
  var->type  = type_;
  return var;
}

Variable Program::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add");
  instr.SetInputs({a, b});
  AddInstruction(instr);
  return instr.GetOutputs()[0];
}

Variable Program::relu(const Variable& a) {
  Instruction instr("relu");
  instr.SetInputs({a});
  AddInstruction(instr);
  return instr.GetOutputs()[0];
}

std::vector<Variable> Program::conv2d(
    const Variable& a,
    const Variable& b,
    const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AddInstruction(instr);
  return instr.GetOutputs();
}

Variable Program::batchnorm(const Variable& a,
                            const Variable& b,
                            const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("batchnorm");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AddInstruction(instr);
  return instr.GetOutputs()[0];
}

Instruction& Program::operator[](size_t i) {
  CHECK_LT(i, instrs.size());
  return instrs[i];
}

const Instruction& Program::operator[](size_t i) const {
  CHECK_LT(i, instrs.size());
  return instrs[i];
}

std::ostream& operator<<(std::ostream& os, const Variable& x) {
  os << "Var(" << x->id << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  std::vector<std::string> outs, ins;
  for (auto& x : instr->inputs) {
    ins.push_back(utils::GetStreamCnt(x));
  }
  for (auto& x : instr->outputs) {
    outs.push_back(utils::GetStreamCnt(x));
  }

  os << utils::Join(outs, ", ") << " = " << instr->op_type << "(" << utils::Join(ins, ", ") << ")";
  return os;
}

}  // namespace frontend
}  // namespace cinn

// CINN_REGISTRY_ENABLE(cinn::hlir::framework::Operator);

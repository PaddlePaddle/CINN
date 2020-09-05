#include "cinn/frontend/syntax.h"

#include "cinn/frontend/paddle/model_parser.h"
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

Instruction::Instruction(std::string_view op_type, const std::vector<Variable>& inputs, Program* parent)
    : common::Shared<_Instruction_>(common::make_shared<_Instruction_>()) {
  get()->op_type        = op_type;
  get()->parent_program = parent;
  get()->inputs         = inputs;
  PrepareOutputs();
}

Placeholder::operator Variable() {
  Variable var(id());
  var->shape = shape();
  var->type  = type_;
  return var;
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

// Add an Instruction to a program given a Paddle-format \p op_desc.
void ProgramAddOp(Program* program, const paddle::cpp::OpDesc& op_desc) {}

void LoadPaddleProgram(const std::string& model_dir, bool is_combined) {
  hlir::framework::Scope scope;
  paddle::cpp::ProgramDesc program_desc;
  paddle::LoadModelPb(model_dir, "__model__", "", &scope, &program_desc, is_combined);
  CHECK_EQ(program_desc.BlocksSize(), 1) << "CINN can only support the model with a single block";
  auto* block_desc = program_desc.GetBlock<paddle::cpp::BlockDesc>(0);
  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
  }
}

Variable Program::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_add(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu(const Variable& a) {
  Instruction instr("relu", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu6(const Variable& a) {
  Instruction instr("relu6", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}
Variable Program::mul(
    const Variable& a, const Variable& b, bool trans_a, bool trans_b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("trans_a", trans_a);
  instr.SetAttr("trans_b", trans_b);
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn

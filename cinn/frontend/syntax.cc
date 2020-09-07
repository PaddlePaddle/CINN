#include "cinn/frontend/syntax.h"

#include <memory>
#include <tuple>
#include <utility>

#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/frontend/paddle_model_to_program.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {
using hlir::framework::Scope;

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

Placeholder::operator Variable() const { return var_; }

Variable Program::conv2d(const Variable& a,
                         const Variable& b,
                         const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(2);
}

Variable Program::batchnorm(const Variable& a,
                            const Variable& b,
                            const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("batchnorm");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::scale(const Variable& a,
                        const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("scale");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::softmax(const Variable& a,
                          const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("softmax");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(1);
}

Instruction& Program::operator[](size_t i) {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
}

const Instruction& Program::operator[](size_t i) const {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
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

std::tuple<std::unique_ptr<Program>,
           std::unordered_map<std::string, Variable>,
           std::unordered_map<std::string, std::string>>
LoadPaddleProgram(const std::string& model_dir, Scope* scope, bool is_combined) {
  LOG(INFO) << "Loading Paddle model from " << model_dir;
  PaddleModelToProgram _(scope);
  return std::make_tuple(_(model_dir, is_combined), _.var_map(), _.var_model_to_program_map());
}

void Program::SetInputs(const std::vector<Variable>& xs) {
  CHECK(!xs.empty()) << "At least one input is needed for a program!";
  for (int i = 0; i < xs.size(); i++) {
    CHECK(!xs[i]->shape.empty()) << "Found " << i << "-th input's shape is not set yet";
    CHECK(!xs[i]->type.is_unk()) << "Found " << i << "-th input's type is not set yet";
    inputs_.push_back(xs[i]);
  }
}

void Program::Validate() const {
  CHECK(!inputs_.empty()) << "Inputs of the program is not set yet";
  CHECK(!instrs_.empty()) << "No instruction is added yet";
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

Variable Program::scale(const Variable& a, float ratio) {
  Instruction instr("scale", {a});
  instr.SetAttr("scale", ratio);
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

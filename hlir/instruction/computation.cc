#include "hlir/instruction/computation.h"
#include <sstream>

namespace hlir {
namespace instruction {

std::string Computation::to_debug_string() const {
  std::stringstream ss;
  ss << "declare " << name_ << " {\n";

  for (const auto &instr : instructions_) {
    ss << "  " << instr->to_debug_string() << "\n";
  }

  ss << "}\n";
  return ss.str();
}

Instruction *Computation::Builder::AddInstruction(std::unique_ptr<Instruction> &&instruction,
                                                  const std::string &comment) {
  instructions_.push_back(std::move(instruction));
  last_added_instruction_ = instructions_.back().get();
  last_added_instruction_->set_id(context_.new_ssa_id());
  if (!comment.empty()) last_added_instruction_->set_comment(comment);
  return last_added_instruction_;
}

std::unique_ptr<Computation> Computation::Builder::Build() {
  CHECK(!is_built_);
  is_built_ = true;
  return std::unique_ptr<Computation>(new Computation(std::move(instructions_), name_));
}

const Shape &Computation::Builder::shape() const {
  CHECK(last_added_instruction_);
  return last_added_instruction_->shape();
}

const type_t &Computation::Builder::type() const {
  CHECK(last_added_instruction_);
  return last_added_instruction_->type();
}

bool Computation::RemoveInstruction(Instruction *instruction) { return false; }

std::vector<Instruction *> Computation::GetParameters() const {
  std::vector<Instruction *> params;
  for (auto &instr : instructions_) {
    if (instr->instr_code() == InstrCode::Parameter) {
      params.push_back(instr.get());
    }
  }
  return params;
}

std::vector<Instruction *> Computation::GetConstants() const {
  std::vector<Instruction *> params;
  for (auto &instr : instructions_) {
    if (instr->instr_code() == InstrCode::Constant) {
      params.push_back(instr.get());
    }
  }
  return params;
}

std::vector<Instruction *> Computation::GetIntermediates() const {
  std::vector<Instruction *> params;
  if (!instructions_.empty()) {
    for (int i = 0; i < instructions_.size() - 1; i++) {
      if (instructions_[i]->instr_code() != InstrCode::Parameter &&
          instructions_[i]->instr_code() != InstrCode::Constant)
        params.push_back(instructions_[i].get());
    }
  }

  return params;
}

std::vector<cinn::Var> Computation::CollectParameters() const {
  std::unordered_set<std::string> var_names;
  std::vector<cinn::Var> vars;
  for (auto &instr : instructions_) {
    for (auto &var : instr->shape().CollectDynamicDims()) {
      if (!var_names.count(var->name)) {
        var_names.insert(var->name);
        vars.push_back(var);
      }
    }
  }
  // TODO(Superjomn) check the parameter offset is sorted.
  return vars;
}

}  // namespace instruction
}  // namespace hlir
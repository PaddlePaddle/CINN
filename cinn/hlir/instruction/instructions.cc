#include "cinn/hlir/instruction/instructions.h"

#include <sstream>

#include "cinn/hlir/instruction/computation.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace instruction {
using cinn::utils::GetStreamCnt;
using cinn::utils::StringFormat;

std::string ParameterInstruction::to_debug_string() {
  std::stringstream ss;
  ss << "%";
  ss << StringFormat("%s :%s%s = parameter(%s)",
                     id().c_str(),
                     GetStreamCnt(type()).c_str(),       //
                     shape().to_debug_string().c_str(),  //
                     shape().to_debug_string().c_str());
  return ss.str();
}

std::string ParameterInstruction::id() const { return name_ + std::to_string(id_); }

std::string CallInstruction::to_debug_string() {
  std::stringstream ss;
  ss << "%" << id() << ": " << type() << shape().to_debug_string();
  ss << " = ";
  ss << computation_->name() << "(";

  std::vector<std::string> vs;
  for (int i = 0; i < operand_count() - 1; i++) {
    vs.push_back("%" + operand(i)->id());
  }

  ss << utils::Join(vs, ", ");

  ss << ")";

  return ss.str();
}

std::unique_ptr<Instruction> Tuple::Get(int i) { return Instruction::CreateTupleGet(this, i); }
std::string Tuple::to_debug_string() {
  std::stringstream ss;
  std::vector<std::string> vs;
  for (int i = 0; i < operand_count(); i++) vs.push_back("%" + operand(i)->id());

  ss << "%" << id() << ":tuple = make_tuple(";
  ss << utils::Join(vs, ", ");
  ss << ")";
  return ss.str();
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

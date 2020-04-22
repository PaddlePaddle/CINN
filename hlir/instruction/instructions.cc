#include "hlir/instruction/instructions.h"
#include <sstream>
#include "cinn/utils/string.h"
#include "hlir/instruction/computation.h"

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
  ss << "%" << id() << " " << type() << shape().to_debug_string();
  ss << " = ";
  ss << computation_->name() << "(";

  if (operand_count() > 0) {
    for (int i = 0; i < operand_count() - 1; i++) {
      ss << operand(i)->id() << ", ";
    }
    ss << operand(operand_count() - 1)->id();
  }

  ss << ")";

  return ss.str();
}
}  // namespace instruction
}  // namespace hlir
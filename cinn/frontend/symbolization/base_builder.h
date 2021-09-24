#pragma once

#include <string>
#include <utility>
#include <vector>

#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace frontend {
namespace symbolization {
class BaseBuilder {
 public:
  using AttrT = hlir::framework::NodeAttr::attr_t;

  explicit BaseBuilder(const std::string& name) : name_(name) {}

  Program Build();

  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");

  // name of this builder
  const std::string& name() { return name_; }

  virtual ~BaseBuilder() {}

 protected:
  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  std::string name_;
  std::vector<Instruction> instrs_;
  std::vector<Variable> inputs_;
};

}  // namespace symbolization
}  // namespace frontend
}  // namespace cinn

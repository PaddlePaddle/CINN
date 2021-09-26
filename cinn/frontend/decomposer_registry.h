#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

using Decomposer = std::function<void(
    const Instruction& instr, frontend::Program* progam, std::unordered_map<std::string, Variable>* outs_map)>;

using TargetMap = std::unordered_map<common::Target, Decomposer, common::Target::Hash>;

class InstrDecomposerMap {
 public:
  static InstrDecomposerMap& Instance() {
    static InstrDecomposerMap g_instr_decomposer_map;
    return g_instr_decomposer_map;
  }

  bool Has(const std::string& op_type, const common::Target& target) const {
    return map_.find(op_type) != map_.end() && map_.at(op_type).find(target) != map_.at(op_type).end();
  }

  void Insert(const std::string& op_type, const common::Target& target, Decomposer func) {
    map_[op_type][target] = func;
  }

  const Decomposer& Get(const std::string& op_type, const common::Target& target) const {
    auto decomposer_ptr = GetNullable(op_type, target);
    CHECK(decomposer_ptr) << "The decomposer for [" << op_type << ", " << target << "] "
                          << " is not registered.";
    return *decomposer_ptr;
  }

  const Decomposer* GetNullable(const std::string& op_type, const common::Target& target) const {
    auto it_target = map_.find(op_type);
    if (it_target == map_.end()) {
      return nullptr;
    }
    auto decomposer_map = it_target->second;
    auto it_decomposer  = decomposer_map.find(target);
    if (it_decomposer == decomposer_map.end()) {
      return nullptr;
    }
    return &it_decomposer->second;
  }

 private:
  std::unordered_map<std::string, TargetMap> map_;
};

class InstrDecomposerRegistry final {
 public:
  static void RegisterDecomposer(const std::string& op_type, const common::Target& target, Decomposer func);
};

#define CINN_REGISTER_INSTR_DECOMPOSER(op_type, target, decomposer) \
  InstrDecomposerRegistry::RegisterDecomposer(op_type, target, decomposer)

}  // namespace frontend
}  // namespace cinn

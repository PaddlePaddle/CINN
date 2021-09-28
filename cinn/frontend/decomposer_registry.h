#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

/*
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
*/

class DecomposerRegistry;

class Decomposer : public Registry<DecomposerRegistry> {
 public:
  /**
   * \brief Get an Decomposer for a given operator name.
   *  Will raise an error if the Decomposer has not been registered.
   * @param op_name Name of the operator.
   * @return Pointer to a Op, valid throughout program lifetime.
   */

  inline const DecomposerRegistry *Find(const std::string &name, const common::Target &target) {
    return Registry<DecomposerRegistry>::Find(name + "_" + target.hash_str());
  }

  inline DecomposerRegistry &__REGISTER__(const std::string &name, const common::Target &target) {
    return Registry<DecomposerRegistry>::__REGISTER__(name + "_" + target.hash_str());
  }

  static Decomposer *Global() {
    static Decomposer inst;
    return &inst;
  }
};

class DecomposerContext {
 private:
  Program program_;
  Instruction instr_;
};

typedef std::function<void(DecomposerContext *context)> DecomposerFunction;

struct DecomposerRegistry : public FunctionRegEntryBase<DecomposerRegistry, DecomposerFunction> {};

#define CINN_DECOMPOSER_REGISTER(name, target)                                              \
  static DecomposerRegistry &CINN_STR_CONCAT(__make_DecomposerRegistry_name, __COUNTER__) = \
      Decomposer::Global()->__REGISTER__(#name, target)

}  // namespace frontend
}  // namespace cinn

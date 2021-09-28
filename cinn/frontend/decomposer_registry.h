#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

class Decomposer;

class DecomposerContext {
 public:
  DecomposerContext(const Instruction& instr, const common::Target& target, Program* program)
      : instr_(instr), target_(target), program_(program) {}

 private:
  const Instruction& instr_;
  const common::Target& target_;
  Program* program_{nullptr};
};

class InstrDecomposerRegistry : public Registry<Decomposer> {
 public:
  InstrDecomposerRegistry() = default;

  static InstrDecomposerRegistry* Global() {
    static InstrDecomposerRegistry x;
    return &x;
  }

  inline const Decomposer* Find(const std::string& name, const common::Target& target) {
    return Registry<Decomposer>::Find(name + "_" + target.hash_str());
  }

  inline Decomposer& __REGISTER__(const std::string& name, const common::Target& target) {
    return Registry<Decomposer>::__REGISTER__(name + "_" + target.hash_str());
  }

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(InstrDecomposerRegistry);
};

class Decomposer {
 public:
  using DecomposerKernel = std::function<void(const DecomposerContext&)>;

  static const Decomposer* Get(const std::string& op_name, const common::Target& target) {
    const Decomposer* decomposer = InstrDecomposerRegistry::Global()->Find(op_name, target);
    CHECK(decomposer) << "Decomposer for [" << op_name << ", " << target << "] is not registered";
    return decomposer;
  }

  Decomposer& Set(const DecomposerKernel& kernel) {
    kernel_ = kernel;
    return *this;
  }

  void Run(const DecomposerContext& context) { kernel_(context); }

  std::string name;

 private:
  // friend class Registry<Decomposer>;
  DecomposerKernel kernel_;
};

#define CINN_DECOMPOSER_REGISTER(name, target)                                                \
  static ::cinn::frontend::Decomposer& CINN_STR_CONCAT(__make_Decomposer_name, __COUNTER__) = \
      ::cinn::frontend::InstrDecomposerRegistry::Global()->__REGISTER__(name, target)

}  // namespace frontend
}  // namespace cinn

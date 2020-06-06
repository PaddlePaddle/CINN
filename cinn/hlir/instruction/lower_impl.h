#pragma once

#include <map>
#include <memory>

#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/module_lower.h"
#include "cinn/hlir/instruction/scope.h"

namespace cinn {
namespace hlir {
namespace instruction {
struct ModuleLower;

/**
 * The base of all the implementation from Lower HLIR instruction to CINN IR.
 * All the implementation should inherient this.
 */
class LowerImplBase {
 public:
  explicit LowerImplBase(InstrCode code) : code_(code) {}

  virtual void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) = 0;

  InstrCode code() const { return code_; }

 private:
  InstrCode code_;
};

/**
 * Registry of all the Lower implementations.
 */
class LowerImplRegistry {
 public:
  using key_t = std::pair<InstrCode, std::string>;

  static LowerImplRegistry& Global();

  void Insert(InstrCode code, const std::string& kind, std::function<std::unique_ptr<LowerImplBase>()>&& creator);

  //! Tell whether this registry contains the impl of (code, kind).
  bool Has(InstrCode code, const std::string& kind) const;

  std::unique_ptr<LowerImplBase> Create(InstrCode code, const std::string& kind);

 private:
  LowerImplRegistry() = default;
  std::map<InstrCode, std::map<std::string, std::function<std::unique_ptr<LowerImplBase>()>>> data_;
};

template <typename T, typename... Args>
struct LowerImplRegistrar {
  LowerImplRegistrar(const std::string& name, InstrCode code, Args... args) {
    LOG(WARNING) << "Register LowerImpl [" << code << ":" << name << "]";
    LowerImplRegistry::Global().Insert(code, name, [=]() -> std::unique_ptr<LowerImplBase> {
      return std::make_unique<T>(code, std::forward(args)...);
    });
  }
  bool Touch() { return true; }
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

#define REGISTER_INSTRUCTION_LOWER(name__, code__, T)                                                               \
  ::cinn::hlir::instruction::LowerImplRegistrar<::cinn::hlir::instruction::primitive::T> name__##code__##registrar( \
      #name__, ::cinn::hlir::instruction::InstrCode::code__);                                                       \
  bool __cinn__TouchInstructionLowerRegistrar_##code__##name__() { return name__##code__##registrar.Touch(); }
#define USE_INSTRUCTION_LOWER(name__, code__)                                     \
  extern bool __cinn__TouchInstructionLowerRegistrar_##code__##name__();          \
  [[maybe_unused]] static bool __cinn__##__##code__##name__##_registrar_touched = \
      __cinn__TouchInstructionLowerRegistrar_##code__##name__();

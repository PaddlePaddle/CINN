#pragma once

#include <map>
#include <memory>

#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/lang/module.h"

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

  virtual void Run(Instruction* instr, cinn::hlir::instruction::Context* context, ModuleLower* module_lower) = 0;

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

  void Insert(InstrCode code, const std::string& kind, std::function<LowerImplBase*()>&& creator);

  LowerImplBase* Create(InstrCode code, const std::string& kind);

 private:
  LowerImplRegistry() = default;
  std::map<InstrCode, std::map<std::string, std::function<LowerImplBase*()>>> data_;
};

template <typename T, typename... Args>
struct LowerImplRegistrar {
  LowerImplRegistrar(const std::string& name, InstrCode code, Args... args) {
    LOG(WARNING) << "Register LowerImpl [" << code << ":" << name << "]";
    LowerImplRegistry::Global().Insert(
        code, name, [=]() -> LowerImplBase* { return new T(code, std::forward(args)...); });
  }
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

namespace std {

namespace ci = cinn::hlir::instruction;  // NOLINT

template <>
struct hash<cinn::hlir::instruction::LowerImplRegistry::key_t> {
  size_t operator()(const ci::LowerImplRegistry::key_t& key) {
    return (static_cast<size_t>(key.first) << 1) ^ (std::hash<std::string>()(key.second));
  }
};

}  // namespace std

#pragma once

#include <map>
#include <string>
#include <utility>

#include "cinn/hlir/instruction/instruction.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace hlir {
namespace instruction {

class ScheduleImplBase {
 public:
  explicit ScheduleImplBase(InstrCode code) : code_(code) {}

  virtual void Run(ir::Tensor* tensor) = 0;

  InstrCode code() const { return code_; }

 private:
  InstrCode code_;
};

/**
 * Registry of all the Schedule implementations.
 */
class ScheduleImplRegistry {
 public:
  using key_t = std::pair<InstrCode, std::string>;

  static ScheduleImplRegistry& Global();

  void Insert(InstrCode code, const std::string& kind, std::function<ScheduleImplBase*()>&& creator);

  ScheduleImplBase* Create(InstrCode code, const std::string& kind);

 private:
  ScheduleImplRegistry() = default;
  std::map<InstrCode, std::map<std::string, std::function<ScheduleImplBase*()>>> data_;
};

template <typename T, typename... Args>
struct ScheduleImplRegistrar {
  ScheduleImplRegistrar(const std::string& name, InstrCode code, Args... args) {
    LOG(WARNING) << "Register ScheduleImpl [" << code << ":" << name << "]";
    ScheduleImplRegistry::Global().Insert(
        code, name, [=]() -> ScheduleImplBase* { return new T(code, std::forward(args)...); });
  }
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

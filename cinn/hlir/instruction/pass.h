#pragma once

#include "cinn/hlir/instruction/module.h"

namespace cinn {
namespace hlir {
namespace instruction {

struct ModuleGroup;

/**
 * The base class of all Passes.
 */
class PassInterface {
 public:
  virtual ~PassInterface()              = default;
  virtual std::string_view name() const = 0;

  virtual bool Run(Module* module)                         = 0;
  virtual bool RunOnModuleGroup(ModuleGroup* module_group) = 0;

  virtual bool is_pass_pipeline() const { return false; }
};

/**
 * The base class of the Passes performed on Modules.
 */
class ModulePass : public PassInterface {
 public:
  bool RunOnModuleGroup(ModuleGroup* module_group) override;
};

/**
 * The base class of the Passes performed on ModuleGroups.
 */
class ModuleGroupPass : public PassInterface {
 public:
  bool Run(Module* module) override { NOT_IMPLEMENTED }
};

class PassRegistry {
 public:
  using creator_t = std::function<std::unique_ptr<PassInterface>()>;

  static PassRegistry& Global() {
    static PassRegistry x;
    return x;
  }

  void Insert(const std::string& name, creator_t&& creator);

  bool Has(const std::string& name) const;

  std::unique_ptr<PassInterface> Create(const std::string& name);
  std::unique_ptr<PassInterface> CreatePromised(const std::string& name);

 private:
  PassRegistry() = default;
  std::map<std::string, std::function<std::unique_ptr<PassInterface>()>> data_;
};

template <typename T, typename... Args>
struct PassRegistrar {
  PassRegistrar(const std::string& name, Args... args) {
    LOG(WARNING) << "Register Pass [" << name << "]";
    PassRegistry::Global().Insert(
        name, [=]() -> std::unique_ptr<PassInterface> { return std::make_unique<T>(name, std::forward(args)...); });
  }
  bool Touch() { return true; }
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

#define REGISTER_PASS(name__, T)                                                                           \
  ::cinn::hlir::instruction::PassRegistrar<::cinn::hlir::instruction::pass::T> name__##registrar(#name__); \
  bool __cinn__TouchPassRegistrar_##name__() { return name__##registrar.Touch(); }
#define USE_PASS(name__)                               \
  extern bool __cinn__##TouchPassRegistrar_##name__(); \
  [[maybe_unused]] static bool __cinn__##name__##_registrar_touched = __cinn__TouchPassRegistrar_##name__();

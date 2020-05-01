#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hlir/instruction/module_group.h"
#include "hlir/instruction/pass.h"
#include "hlir/instruction/pass_pipeline.h"

namespace hlir {
namespace instruction {

/**
 * A pipeline of Passes, it behaves like a single Pass.
 */
class PassPipeline : public PassInterface {
 public:
  explicit PassPipeline(const std::string& name) : name_(name) {}

  std::string_view name() const override { return name_; }

  template <typename T, typename... Args>
  T& AddPass(Args&&... args) {
    CHECK(!run_called_) << "AddPass cannot be called after Run";
    passes_.emplace_back(std::unique_ptr<PassInterface>(new T(std::forward<Args>(args)...)));
    return *static_cast<T*>(passes_.back().get());
  }

  bool Run(Module* module) override;

  bool RunOnModuleGroup(ModuleGroup* module_group) override;

 private:
  std::string name_;
  mutable bool run_called_{false};
  std::vector<std::unique_ptr<PassInterface>> passes_;
};

}  // namespace instruction
}  // namespace hlir

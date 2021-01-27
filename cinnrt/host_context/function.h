#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <string>
#include "cinn/common/macros.h"
#include "cinnrt/host_context/value.h"

namespace cinnrt {
namespace host_context {

/**
 * Base class of all executable Function.
 *
 * This is used by `cinn.call` op, to execute a function.
 */
class Function {
 public:
  CINN_DISALLOW_COPY_AND_ASSIGN(Function);

  std::string_view name() const { return name_; }

  size_t num_arguments() const { return num_arguments_; }
  size_t num_results() const { return num_results_; }

  virtual void Execute(llvm::ArrayRef<Value*> arguments, llvm::MutableArrayRef<ValueRef> results) const = 0;

  virtual ~Function() = default;

 protected:
  Function(std::string_view name, size_t num_arguments, size_t num_results)
      : name_(name), num_arguments_(num_arguments), num_results_(num_results) {}

 private:
  std::string name_;
  size_t num_arguments_{};
  size_t num_results_{};
};

}  // namespace host_context
}  // namespace cinnrt

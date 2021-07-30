#pragma once
#include <llvm/ADT/ArrayRef.h>

#include <string>

namespace cinnrt {
namespace host_context {

struct Value;
struct ValueRef;

/**
 * Base class of all executable Function.
 *
 * This is used by `cinn.call` op, to execute a function.
 */
class Function {
 public:
  Function(Function&& other)
      : name_(other.name_), num_arguments_(other.num_arguments_), num_results_(other.num_results_) {}

  Function() = delete;

  std::string name() const { return name_; }

  size_t num_arguments() const { return num_arguments_; }
  size_t num_results() const { return num_results_; }

  virtual void Execute(llvm::ArrayRef<Value*> arguments,
                       llvm::MutableArrayRef<ValueRef> results,
                       bool is_region = false) const {}

  virtual ~Function() = default;

 protected:
  Function(std::string name, size_t num_arguments, size_t num_results)
      : name_(name), num_arguments_(num_arguments), num_results_(num_results) {}

 private:
  std::string name_;
  size_t num_arguments_{};
  size_t num_results_{};
};

}  // namespace host_context
}  // namespace cinnrt

#pragma once
#include <string>
#include <vector>

#include "cinn/backends/outputs.h"
#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/function.h"

namespace cinn {
namespace lang {

class _Module_;

/**
 * Module represents IR containing lowered function definitions and buffers.
 */
class Module {
 public:
  Module(const std::string& name, const Target& target);

  //! Get the target of this module.
  const Target& target() const;

  //! Get the name of the module.
  const std::string& name() const;

  //! The members in the module.
  // @{
  const std::vector<ir::Buffer>& buffers() const;
  const std::vector<ir::PackedFunc>& functions() const;
  const std::vector<Module>& submodules() const;
  // @}

  //! Add something to this module.
  // @{
  void Append(const ir::Buffer& buffer);
  void Append(const ir::PackedFunc& function);
  void Append(const Module& module);
  // @}

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

  _Module_* self();
  const _Module_* self() const;

 private:
  Shared<_Module_> module_;
};

/**
 * A struct representing an argument to a lowered function. Used for specifying the function signature of generated
 * code.
 */
struct Argument {
  //! The name of the argument.
  std::string name;

  enum class Kind { kScalar = 0, kBuffer } kind{Kind::kScalar};

  //! Number of the dimensions of buffer.
  uint32_t ndims{0};

  //! The type of the buffer or scalar.
  Type type;

  bool is_buffer() const { return kind == Kind::kBuffer; }
  bool is_scalar() const { return kind == Kind::kScalar; }

  Argument() {}
  Argument(const std::string& name, Kind kind, const Type& type, int ndims)
      : name(name), kind(kind), type(type), ndims(ndims) {}

  explicit Argument(const ir::Buffer& buffer) : name(buffer->name), type(buffer->type()), ndims(buffer->shape.size()) {}
};

/**
 * Definition of a lowered function. Note that, it should be functional.
 */
struct LoweredFunc {
  //! The name of this function.
  std::string name;

  //! The Arguments used in the body of the function.
  std::vector<Argument> args;

  //! Body of this function.
  Expr body;

  LoweredFunc(const std::string& name, const std::vector<Argument>& args, const Expr& body)
      : name(name), args(args), body(body) {}
};

}  // namespace lang
}  // namespace cinn

#pragma once
#include <string>

namespace cinn {
namespace backends {

/**
 * A struct specifying a collection of outputs.
 */
struct Outputs {
  //! The name of the emitted object file. Empty if no object file is desired.
  std::string object_name;

  //! The name of the emitted llvm bitcode. Empty if no bitcode file is desired.
  std::string bitcode_name;

  //! The name of the emitted C header file.
  std::string c_header_name;

  //! The name of the emitted C source file.
  std::string c_source_name;

  Outputs object(const std::string& name) const;

  Outputs bitcode(const std::string& name) const;

  Outputs c_header(const std::string& name) const;

  Outputs c_source(const std::string& name) const;
};

}  // namespace backends
}  // namespace cinn

#include "cinn/backends/outputs.h"

namespace cinn {
namespace lang {}  // namespace lang

backends::Outputs backends::Outputs::object(const std::string &name) const {
  Outputs updated     = *this;
  updated.object_name = name;
  return updated;
}

backends::Outputs backends::Outputs::bitcode(const std::string &name) const {
  Outputs updated      = *this;
  updated.bitcode_name = name;
  return updated;
}

backends::Outputs backends::Outputs::c_header(const std::string &name) const {
  Outputs updated       = *this;
  updated.c_header_name = name;
  return updated;
}

backends::Outputs backends::Outputs::c_source(const std::string &name) const {
  Outputs updated       = *this;
  updated.c_source_name = name;
  return updated;
}

}  // namespace cinn

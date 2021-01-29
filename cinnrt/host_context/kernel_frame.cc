#include "cinnrt/host_context/kernel_frame.h"

#include <memory>

namespace cinnrt {
namespace host_context {

std::ostream& operator<<(std::ostream& os, const KernelFrame& frame) {
  os << "KernelFrame: " << frame.GetNumArgs() << " args, " << frame.GetNumResults() << " res, " << frame.GetNumResults()
     << " attrs";
  return os;
}

}  // namespace host_context
}  // namespace cinnrt

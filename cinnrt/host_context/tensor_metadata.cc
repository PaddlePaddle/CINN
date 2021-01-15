#include "cinnrt/host_context/tensor_metadata.h"
#include <llvm/Support/raw_ostream.h>

namespace cinnrt {
namespace host_context {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorMetadata& meta) {
  os << meta.dtype.name();
  os << "\n";
  os << meta.shape;
}

}  // namespace host_context
}  // namespace cinnrt
#include "infrt/tensor/tensor_metadata.h"

#include <llvm/Support/raw_ostream.h>

namespace infrt {
namespace tensor {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorMetadata& meta) {
  os << meta.dtype.name();
  os << "\n";
  os << meta.shape;
}

}  // namespace tensor
}  // namespace infrt

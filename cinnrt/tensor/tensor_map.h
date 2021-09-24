#include <string>
#include <absl/container/flat_hash_map.h>

#include "cinnrt/tensor/dense_host_tensor.h"

namespace cinnrt {
namespace tensor {

using TensorMap = absl::flat_hash_map<std::string, tensor::DenseHostTensor*>;

TensorMap* LoadParams(const std::string& path);

}  // namespace tensor
}  // namespace cinnrt

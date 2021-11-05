#include <absl/container/flat_hash_map.h>

#include <string>

#include "infrt/tensor/dense_host_tensor.h"

namespace infrt {
namespace tensor {

using TensorMap = absl::flat_hash_map<std::string, tensor::DenseHostTensor*>;

TensorMap* LoadParams(const std::string& path);

}  // namespace tensor
}  // namespace infrt

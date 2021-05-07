#include <string>
#include <unordered_map>

#include "cinnrt/tensor/dense_host_tensor.h"

namespace cinnrt {
namespace tensor {

using TensorMap = std::unordered_map<std::string, tensor::DenseHostTensor*>;

TensorMap* LoadParams(const std::string& path);

}  // namespace tensor
}  // namespace cinnrt

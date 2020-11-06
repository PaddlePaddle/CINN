#pragma once

#include <memory>

#include "cinn/common/shared.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinn::hlir::framework {
class Buffer;
}  // namespace cinn::hlir::framework

namespace cinn::host_context {

enum class DeviceKind {
  kCPU = 0,
};

// TODO(Superjomn) Replace the hlir/framework/Tensor with this.
/**
 * DenseTensor is a dense tensor, it holds a TensorShape and a buffer.
 */
class DenseTensor {
 public:
  DenseTensor(const TensorShape& shape, const cinn_type_t& dtype, DeviceKind device = DeviceKind::kCPU);

  const TensorShape& shape() const;

  const hlir::framework::Buffer* buffer() const;

  void* data() const;

  friend std::ostream& operator<<(std::ostream& os, const DenseTensor& instance);

  ~DenseTensor();

 private:
  TensorShape shape_;
  cinn_type_t dtype_;
  std::shared_ptr<hlir::framework::Buffer> buffer_;
};

}  // namespace cinn::host_context

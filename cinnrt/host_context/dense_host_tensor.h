#pragma once

#include <memory>
#include <utility>

#include "cinnrt/host_context/tensor_metadata.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinn::hlir::framework {
class Buffer;
}  // namespace cinn::hlir::framework

namespace cinnrt::host_context {

enum class DeviceKind {
  kCPU = 0,
};

class Tensor {
 public:
  virtual bool IsHostTensor() const = 0;

  const TensorMetadata& metadata() const { return metadata_; }

 protected:
  explicit Tensor(const TensorMetadata& metadata) : metadata_(metadata) {}
  explicit Tensor(TensorMetadata&& metadata) : metadata_(std::move(metadata)) {}

 private:
  TensorMetadata metadata_;
};

class HostTensor : public Tensor {
 public:
  bool IsHostTensor() const override { return true; }

 protected:
  explicit HostTensor(const TensorMetadata& metadata) : Tensor(metadata) {}
  explicit HostTensor(TensorMetadata&& metadata) : Tensor(std::move(metadata)) {}
};

// TODO(Superjomn) Replace the hlir/framework/Tensor with this.
/**
 * DenseTensor is a dense tensor, it holds a TensorShape and a buffer.
 */
class DenseHostTensor : public HostTensor {
 public:
  DenseHostTensor(const TensorShape& shape, DType dtype);

  const TensorShape& shape() const;

  const cinn::hlir::framework::Buffer* buffer() const;

  void* raw_data() const;

  friend std::ostream& operator<<(std::ostream& os, const DenseHostTensor& instance);

  ~DenseHostTensor();

 private:
  // TODO(Superjomn) Discard the dependency of the Buffer in cinncore or create a general buffer in common.
  std::shared_ptr<cinn::hlir::framework::Buffer> buffer_;
};

}  // namespace cinnrt::host_context

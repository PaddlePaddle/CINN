#pragma once

#include <memory>
#include <utility>

#include "cinnrt/tensor/tensor_metadata.h"
#include "cinnrt/tensor/tensor_shape.h"

namespace cinn::hlir::framework {
class Buffer;
}  // namespace cinn::hlir::framework

namespace cinnrt::tensor {

enum class DeviceKind {
  kCPU = 0,
};

class Tensor {
 public:
  virtual bool IsHostTensor() const = 0;
  virtual ~Tensor()                 = default;

  const TensorMetadata& metadata() const { return metadata_; }

 protected:
  Tensor() = default;
  void setTensorMetadata(TensorMetadata& metadata) { metadata_ = metadata; }
  explicit Tensor(const TensorMetadata& metadata) : metadata_(metadata) {}
  explicit Tensor(TensorMetadata&& metadata) : metadata_(std::move(metadata)) {}

 private:
  TensorMetadata metadata_;
};

class HostTensor : public Tensor {
 public:
  bool IsHostTensor() const override { return true; }

 protected:
  HostTensor() = default;
  explicit HostTensor(const TensorMetadata& metadata) : Tensor(metadata) {}
  explicit HostTensor(TensorMetadata&& metadata) : Tensor(std::move(metadata)) {}
};

// TODO(Superjomn) Replace the hlir/framework/Tensor with this.
/**
 * DenseTensor is a dense tensor, it holds a TensorShape and a buffer.
 */
class DenseHostTensor : public HostTensor {
 public:
  DenseHostTensor() = default;
  DenseHostTensor(const TensorShape& shape, DType dtype);

  void Init(const std::vector<int64_t>& shape, DType dtype);
  const TensorShape& shape() const;

  const cinn::hlir::framework::Buffer* buffer() const;

  void* raw_data() const;

  friend std::ostream& operator<<(std::ostream& os, const DenseHostTensor& instance);

  ~DenseHostTensor();

 private:
  // TODO(Superjomn) Discard the dependency of the Buffer in cinncore or create a general buffer in common.
  std::shared_ptr<cinn::hlir::framework::Buffer> buffer_;
};

}  // namespace cinnrt::tensor

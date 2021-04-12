#include "cinnrt/host_context/value.h"

#include "cinnrt/tensor/dense_tensor_view.h"

namespace cinnrt {
namespace host_context {

ValueRef::ValueRef(int32_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(int64_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(float val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(double val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(bool val) : Shared<Value>(new Value(val)) {}

const char* Value::type_info() const { return __type_info__; }

void CopyTo(const Value& from, Value* to) {
  CHECK(from.valid()) << "from value is not valid, can't be copied";
  CHECK(to) << "to is not valid";
  visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int16_t>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, int32_t>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, float>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, double>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, uint32_t>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, uint64_t>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, bool>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, tensor::TensorShape>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, MlirFunctionExecutable*>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, tensor::DenseHostTensor>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, std::vector<int16_t>>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
          to->data = arg;
        else if constexpr (std::is_same_v<T, tensor::TensorMap>)
          to->data = arg;
        else
          LOG(FATAL) << "Not supported Value copy: " << typeid(T).name();
      },
      from.data);
}

}  // namespace host_context
}  // namespace cinnrt

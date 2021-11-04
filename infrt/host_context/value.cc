#include "infrt/host_context/value.h"

#include "infrt/tensor/dense_tensor_view.h"

namespace infrt {
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
        if (std::is_same<T, int16_t>::value)
          to->data = arg;
        else if (std::is_same<T, int32_t>::value)
          to->data = arg;
        else if (std::is_same<T, float>::value)
          to->data = arg;
        else if (std::is_same<T, double>::value)
          to->data = arg;
        else if (std::is_same<T, uint32_t>::value)
          to->data = arg;
        else if (std::is_same<T, uint64_t>::value)
          to->data = arg;
        else if (std::is_same<T, bool>::value)
          to->data = arg;
        else if (std::is_same<T, tensor::TensorShape>::value)
          to->data = arg;
        else if (std::is_same<T, MlirFunctionExecutable*>::value)
          to->data = arg;
        else if (std::is_same<T, tensor::DenseHostTensor>::value)
          to->data = arg;
        else if (std::is_same<T, std::vector<int16_t>>::value)
          to->data = arg;
        else if (std::is_same<T, std::vector<int64_t>>::value)
          to->data = arg;
        else if (std::is_same<T, tensor::TensorMap>::value)
          to->data = arg;
        else
          LOG(FATAL) << "Not supported Value copy: " << typeid(T).name();
      },
      from.data);
}

}  // namespace host_context
}  // namespace infrt

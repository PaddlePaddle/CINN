#include "cinnrt/host_context/value.h"

#include <variant>

#include "cinnrt/host_context/dense_tensor_view.h"

namespace cinn {
namespace host_context {

ValueRef::ValueRef(int32_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(int64_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(float val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(double val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(bool val) : Shared<Value>(new Value(val)) {}

const char* Value::type_info() const { return __type_info__; }

}  // namespace host_context
}  // namespace cinn

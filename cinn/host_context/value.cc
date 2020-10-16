#include "cinn/host_context/value.h"
#include <variant>

namespace cinn {
namespace host_context {

Value::Value(int32_t val) : Shared<_Value_>(new _Value_(val)) {}
Value::Value(int64_t val) : Shared<_Value_>(new _Value_(val)) {}
Value::Value(float val) : Shared<_Value_>(new _Value_(val)) {}
Value::Value(double val) : Shared<_Value_>(new _Value_(val)) {}
Value::Value(bool val) : Shared<_Value_>(new _Value_(val)) {}

const char* _Value_::type_info() const { return __type_info__; }

}  // namespace host_context
}  // namespace cinn

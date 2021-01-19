#include "cinnrt/common/dtype.h"

namespace cinnrt {

const char* DType::name() const {
  switch (kind_) {
#define CINNRT_DTYPE(enum__, value__) \
  case Kind::enum__:                  \
    return #enum__;                   \
    break;
#include "cinnrt/common/dtype.def"
#undef CINNRT_DTYPE
  }

  return "";
}

size_t DType::GetHostSize() const {
  switch (kind_) {
#define CINNRT_DTYPE(enum__, value__) \
  case DType::Kind::enum__:           \
    return sizeof(DTypeInternal<DType::Kind::enum__>::type);
#include "cinnrt/common/dtype.def"
#undef CINNRT_DTYPE
  }
  return 0;
}

}  // namespace cinnrt
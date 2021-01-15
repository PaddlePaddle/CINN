#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace cinnrt {
class DType {
 public:
  enum class Kind : uint8_t {
    Unk = 0,

  // Automatically generate the enum definition
#define CINNRT_DTYPE(enum__, value__) enum__ = value__,
#include "cinnrt/common/dtype.def"
#undef CINNRT_DTYPE

    BOOL = I1,
  };

  explicit constexpr DType(Kind kind) : kind_(kind) {}

  DType(const DType&) = default;
  DType& operator=(const DType&) = default;
  bool operator==(DType other) const { return kind_ == other.kind_; }
  bool operator!=(DType other) const { return !(*this == other); }

  constexpr Kind kind() const { return kind_; }

  bool is_valid() const { return kind_ == Kind::Unk; }

  const char* name() const;

  size_t GetHostSize() const;

 private:
  Kind kind_{Kind::Unk};
};

template <typename T>
constexpr DType GetDType();

template <DType::Kind kind>
struct DTypeInternal;

#define CINNRT_IMPL_GET_DTYPE(cpp_type__, enum__) \
  template <>                                     \
  inline constexpr DType GetDType<cpp_type__>() { \
    return DType{DType::Kind::enum__};            \
  }                                               \
  template <>                                     \
  struct DTypeInternal<DType::Kind::enum__> {     \
    using type = cpp_type__;                      \
  };

CINNRT_IMPL_GET_DTYPE(bool, I1);
CINNRT_IMPL_GET_DTYPE(int8_t, I8);
CINNRT_IMPL_GET_DTYPE(int16_t, I16);
CINNRT_IMPL_GET_DTYPE(int32_t, I32);
CINNRT_IMPL_GET_DTYPE(int64_t, I64);
CINNRT_IMPL_GET_DTYPE(uint8_t, UI8);
CINNRT_IMPL_GET_DTYPE(uint16_t, UI16);
CINNRT_IMPL_GET_DTYPE(uint32_t, UI32);
CINNRT_IMPL_GET_DTYPE(uint64_t, UI64);
CINNRT_IMPL_GET_DTYPE(float, F32);
CINNRT_IMPL_GET_DTYPE(double, F64);
CINNRT_IMPL_GET_DTYPE(std::string, STRING);

}  // namespace cinnrt

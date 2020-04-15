#pragma once
/**
 * \file This file implements some intrinsic types used in CodeGen.
 */

#include "cinn/common/common.h"

namespace cinn {
namespace runtime {

/**
 * Type representation for cinn_buffer_t.
 */
struct BufferType {
  static BufferType Create(const Type& primitive) { return BufferType(primitive); }

  static Type cinn_type();

 private:
  explicit BufferType(const Type& primitive_type) : primitive_type(primitive_type) {
    CHECK(primitive_type.valid());
    CHECK(primitive_type.is_primitive());
  }

  //! Determine the primitive of cinn_buffer_t.
  Type primitive_type;
  static char c_type_repr[];
};

static Type make_intrinsic_buffer_type(Type primitive_type) {
  CHECK(primitive_type.is_primitive());
  CHECK(primitive_type.valid());
  Type res = BufferType::cinn_type();
  return res;
}

}  // namespace runtime
}  // namespace cinn

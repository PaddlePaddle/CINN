#pragma once
#include <cstring>

#include "cinn/common/shared.h"

namespace cinn {
namespace common {

template <typename T>
class Shared;
/**
 * Object is the basic element in the CINN, with `Shared` wrapper, the object can be shared accross the system.
 */
struct Object {
  //! Get the type representation of this object.
  virtual const char* type_info() const = 0;

  //! Cast to a derived type.
  template <typename T>
  T* as() {
    return static_cast<T*>(this);
  }

  //! Cast to a derived type.
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(this);
  }

  //! Type safe cast.
  template <typename T>
  T* safe_as() {
    CHECK(std::strcmp(type_info(), T::__type_info__) == 0)
        << "type mismatch, this is a " << type_info() << ", but want a " << T::__type_info__;
    return static_cast<T*>(this);
  }
  //! Type safe cast.
  template <typename T>
  const T* safe_as() const {
    CHECK(std::strcmp(type_info(), T::__type_info__) == 0)
        << "type mismatch, this is a " << type_info() << ", but want a " << T::__type_info__;
    return static_cast<const T*>(this);
  }

  //! Check if the type is right.
  template <typename T>
  bool check_type() const {
    if (std::strcmp(type_info(), T::__type_info__) == 0) {
      return true;
    }
    return false;
  }

  //! The reference count, which make all the derived type able to share.
  mutable RefCount __ref_count__;
};

using object_ptr    = Object*;
using shared_object = Shared<Object>;

}  // namespace common
}  // namespace cinn

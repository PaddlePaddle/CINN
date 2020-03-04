#pragma once
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

  //! The reference count, which make all the derived type able to share.
  mutable RefCount __ref_count__;
};

using object_ptr    = Object*;
using shared_object = Shared<Object>;

}  // namespace common
}  // namespace cinn

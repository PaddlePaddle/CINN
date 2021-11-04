#pragma once

#include <absl/strings/string_view.h>

#include "infrt/common/axis.h"
#include "infrt/common/macros.h"
#include "infrt/common/shared.h"
#include "infrt/common/target.h"
#include "infrt/common/type.h"

namespace infrt {

// export some general concepts.
using common::make_shared;
using common::Object;
using common::ref_count;
using common::Shared;

// Type related.
using common::Bool;
using common::Float;
using common::Int;
using common::UInt;
using common::Void;

using common::type_of;

using common::Target;
using common::Type;
using common::UnkTarget;

template <typename T>
T& Reference(const T* x) {
  return *const_cast<T*>(x);
}

static void CheckVarNameValid(const absl::string_view name) {
  CHECK(!name.empty());
  CHECK(name.find(' ') == std::string::npos &&   //
        name.find('.') == std::string::npos &&   //
        name.find('/') == std::string::npos &&   //
        name.find('\t') == std::string::npos &&  //
        name.find('\n') == std::string::npos &&  //
        name.find('\r') == std::string::npos)
      << "Some invalid character found";
  CHECK(!infrt::common::IsAxisNameReserved(std::string(name)))
      << "The name [" << name << "] is reserved for internal axis";
}

}  // namespace infrt

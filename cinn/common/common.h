#pragma once

#include "cinn/common/axis.h"
#include "cinn/common/cinn_value.h"
#include "cinn/common/context.h"
#include "cinn/common/graph_utils.h"
#include "cinn/common/macros.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"

namespace cinn {

// export some general concepts.
using common::Context;
using common::make_shared;
using common::Object;
using common::ref_count;
using common::Shared;
using common::UniqName;

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

static void CheckVarNameValid(const std::string_view name) {
  CHECK(!name.empty());
  CHECK(name.find(' ') == std::string::npos &&   //
        name.find('.') == std::string::npos &&   //
        name.find('/') == std::string::npos &&   //
        name.find('\t') == std::string::npos &&  //
        name.find('\n') == std::string::npos &&  //
        name.find('\r') == std::string::npos)
      << "Some invalid character found";
  CHECK(!common::IsAxisNameReserved(std::string(name))) << "The name [" << name << "] is reserved for internal axis";
}

}  // namespace cinn

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

}  // namespace cinn

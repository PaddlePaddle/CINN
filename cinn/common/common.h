#pragma once

#include "cinn/common/context.h"
#include "cinn/common/domain.h"
#include "cinn/common/graph_utils.h"
#include "cinn/common/pod_value.h"
#include "cinn/common/shared.h"
#include "cinn/common/type.h"

namespace cinn {

// export some general concepts.
using common::make_shared;
using common::Object;
using common::Shared;
using common::Context;

// Type related.
using common::Float;
using common::Int;
using common::type_of;

}  // namespace cinn

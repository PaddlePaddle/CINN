#pragma once
//! This file includes some utilities imported from  LLVM.
#include "llvm/Support/Error.h"

namespace cinn::utils {

template <typename T>
using Expected = llvm::Expected<T>;

}  // namespace cinn::utils

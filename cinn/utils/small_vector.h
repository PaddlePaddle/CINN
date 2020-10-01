#pragma once
#include "llvm/ADT/SmallVector.h"

namespace cinn::utils {

template <typename Type, unsigned Num>
using SmallVector = llvm::SmallVector<Type, Num>;

}  // namespace cinn::utils

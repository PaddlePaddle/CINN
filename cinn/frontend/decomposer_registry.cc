#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {

void InstrDecomposerRegistry::RegisterDecomposer(const std::string& op_type,
                                                 const common::Target& target,
                                                 Decomposer func) {
  InstrDecomposerMap::Instance().Insert(op_type, target, func);
}

}  // namespace frontend
}  // namespace cinn

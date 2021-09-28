#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {
namespace decomposer {
void relu(const DecomposerContext& context) {}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(decompose) {
  CINN_DECOMPOSER_REGISTER("relu", ::cinn::common::DefaultNVGPUTarget()).Set(cinn::frontend::decomposer::relu);

  return true;
}

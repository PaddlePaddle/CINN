#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {

void decompose_conv(DecomposerContext* context) {}

CINN_DECOMPOSER_REGISTER("conv", common::DefaultHostTarget()).set_body(decompose_conv);
CINN_DECOMPOSER_REGISTER("conv", common::DefaultNVGPUTarget()).set_body(decompose_conv);

}  // namespace frontend
}  // namespace cinn

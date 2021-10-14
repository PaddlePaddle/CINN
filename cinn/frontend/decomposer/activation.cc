
#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {
namespace decomposer {

void relu(const Instruction& instr, const DecomposerContext& context) { LOG(FATAL) << "not implemented"; }

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(activation) {
  CINN_DECOMPOSER_REGISTER(relu, ::cinn::common::DefaultHostTarget()).set_body(cinn::frontend::decomposer::relu);

  return true;
}

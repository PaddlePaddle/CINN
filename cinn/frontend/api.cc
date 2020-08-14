#include "cinn/frontend/api.h"

namespace cinn {
namespace frontend {

Variable add(Variable a, Variable b) {
  Instruction instr("add");
  instr.SetInputs({a, b});
  return instr.GetOutputs()[0];
}

}  // namespace frontend
}  // namespace cinn

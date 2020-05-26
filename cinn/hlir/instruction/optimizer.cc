#include "cinn/hlir/instruction/optimizer.h"

#include "cinn/hlir/instruction/pass/buffer_assign_pass.h"
#include "cinn/hlir/instruction/pass/display_program.h"

namespace cinn {
namespace hlir {
namespace instruction {

Optimizer::Optimizer() : pass_pipeline_("optimizer-pass-pipeline") {
  pass_pipeline_.AddPass<pass::DisplayProgram>("display-program");
  pass_pipeline_.AddPass<pass::BufferAssignPass>("buffer-assign");
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

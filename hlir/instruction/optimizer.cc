#include "hlir/instruction/optimizer.h"

#include "hlir/instruction/pass/display_program.h"

namespace hlir {
namespace instruction {

Optimizer::Optimizer() : pass_pipeline_("optimizer-pass-pipeline") {
  pass_pipeline_.AddPass<pass::DisplayProgram>("display-program");
}

}  // namespace instruction
}  // namespace hlir

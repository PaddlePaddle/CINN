#include "cinn/hlir/instruction/optimizer.h"

#include "cinn/hlir/instruction/pass/use_passes.h"

namespace cinn {
namespace hlir {
namespace instruction {

Optimizer::Optimizer() : pass_pipeline_("optimizer-pass-pipeline") {
  pass_pipeline_.AddPass(PassRegistry::Global().CreatePromised("display_program"));
  pass_pipeline_.AddPass(PassRegistry::Global().CreatePromised("buffer_assign"));
  pass_pipeline_.AddPass(PassRegistry::Global().CreatePromised("lower_kind_determine"));
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

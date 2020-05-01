#pragma once
#include "hlir/instruction/pass_pipeline.h"

namespace hlir {
namespace instruction {

class Optimizer {
 public:
  Optimizer();

  void Run(Module* module) { pass_pipeline_.Run(module); }

 private:
  PassPipeline pass_pipeline_;
};

}  // namespace instruction
}  // namespace hlir

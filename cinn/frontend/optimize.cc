// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/frontend/optimize.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"

DECLARE_bool(cinn_use_new_fusion_pass);

namespace cinn {
namespace frontend {

OptimizeOptions DefaultTrainingOptimizeOptions() {
  OptimizeOptions options;
  options.program_passes = {
      "Decomposer", "TransposeFolding", "GemmRewriter", "ReshapeRewriter", "RemoveIdentity", "DeadCodeEliminate"};
  if (FLAGS_cinn_use_new_fusion_pass) {
    options.graph_passes = {"OpFusionPass", "FusionMergePass"};
  } else {
    options.graph_passes = {"OpFusion"};
  }
  return options;
}

std::shared_ptr<hlir::framework::Graph> Optimize(frontend::Program* program,
                                                 const std::unordered_set<std::string>& fetch_ids,
                                                 common::Target target,
                                                 const OptimizeOptions& options) {
  // Apply program passes
  frontend::ProgramPass::Apply(program, fetch_ids, target, options.program_passes);
  // Apply graph passes
  auto graph = std::make_shared<hlir::framework::Graph>(*program, target);
  hlir::framework::ApplyPasses(graph.get(), options.graph_passes);
  return graph;
}
}  // namespace frontend
}  // namespace cinn

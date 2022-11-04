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

DECLARE_bool(cinn_open_fusion_optimize);
DECLARE_bool(cinn_use_new_fusion_pass);
DECLARE_bool(cinn_use_fill_constant_folding);
DECLARE_bool(cinn_check_fusion_accuracy_pass);

namespace cinn {
namespace frontend {

OptimizeOptions DefaultTrainingOptimizeOptions() {
  OptimizeOptions options;
  options.program_passes.emplace_back("Decomposer");
  options.program_passes.emplace_back("TransposeCollapsing");
  options.program_passes.emplace_back("TransposeFoldingInput");
  options.program_passes.emplace_back("GemmRewriter");
  options.program_passes.emplace_back("TransposeFoldingOutput");
  options.program_passes.emplace_back("GemmRewriter");
  options.program_passes.emplace_back("ReshapeRewriter");
  if (FLAGS_cinn_use_fill_constant_folding) {
    options.program_passes.emplace_back("FillConstantFolding");
  }
  options.program_passes.emplace_back("RemoveIdentity");
  options.program_passes.emplace_back("DeadCodeEliminate");
  if (FLAGS_cinn_open_fusion_optimize) {
    if (FLAGS_cinn_use_new_fusion_pass) {
      options.graph_passes = {
#ifdef CINN_WITH_CUDA
          "MatmulToCublasCustomCallPass",
#ifdef CINN_WITH_CUDNN
          "ConvToCudnnCustomCallPass",
#endif
#endif
          "OpFusionPass",
          "FusionMergePass"};
    } else {
      options.graph_passes = {"OpFusion"};
    }
  }

  // WARNING: the pass must be the last pass !!!
  if (FLAGS_cinn_check_fusion_accuracy_pass) {
    // Check the correct of fusion kernels, if the results not satisfied 'allclose(rtol=1e-05f, atol=1e-08f)', report
    // error and exited.
    options.graph_passes.emplace_back("CheckFusionAccuracyPass");
  }

  return options;
}

std::vector<std::string> DefaultOpFusionPasses() {
  std::vector<std::string> passes;
  if (FLAGS_cinn_open_fusion_optimize) {
    if (FLAGS_cinn_use_new_fusion_pass) {
      passes = {"OpFusionPass", "FusionMergePass"};
    } else {
      passes = {"OpFusion"};
    }
  }
  return passes;
}

std::shared_ptr<hlir::framework::Graph> Optimize(frontend::Program* program,
                                                 const std::unordered_set<std::string>& fetch_ids,
                                                 common::Target target,
                                                 const OptimizeOptions& options) {
  // Apply program passes
  VLOG(3) << "Before frontend::ProgramPass::Apply";
  frontend::ProgramPass::Apply(program, fetch_ids, target, options.program_passes);
  // Apply graph passes
  auto graph = std::make_shared<hlir::framework::Graph>(*program, fetch_ids, target);
  //
  VLOG(3) << "Before hlir::framework::ApplyPasses";
  hlir::framework::ApplyPasses(graph.get(), options.graph_passes);
  return graph;
}
}  // namespace frontend
}  // namespace cinn

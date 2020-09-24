#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

/**
 * The executor for a model.
 */
class Executor final {
 public:
  Executor(const std::vector<std::string>& input_names, const std::vector<hlir::framework::shape_t>& input_shapes)
      : scope_(std::make_shared<hlir::framework::Scope>()), input_names_(input_names), input_shapes_(input_shapes) {}

  /**
   * Load a Paddle model.
   * @param model_dir The directory path to the model.
   * @param params_combined Whether the parameters are composed to a single file.
   */
  void LoadPaddleModel(const std::string& model_dir, bool params_combined = false);

  /**
   * Run the executor.
   */
  void Run();

  hlir::framework::Tensor GetTensor(const std::string& name);

  std::shared_ptr<hlir::framework::Scope> scope();

 private:
  /**
   * Build the model.
   * @param input_names The name of input variables.
   * @param input_shapes The input shapes.
   */
  void Build(const std::vector<std::string>& input_names, const std::vector<hlir::framework::shape_t>& input_shapes);

  std::vector<std::string> input_names_;
  std::vector<hlir::framework::shape_t> input_shapes_;

  std::shared_ptr<hlir::framework::Scope> scope_;
  std::unique_ptr<frontend::Program> program_;
  std::unique_ptr<hlir::framework::GraphCompiler> graph_compiler_;

  std::unordered_map<std::string, Variable> var_map_;
  std::unordered_map<std::string, std::string> var_map_paddle_to_cinn_;
  std::unordered_map<std::string, std::string> var_map_cinn_to_paddle_;

  std::unique_ptr<hlir::framework::Program> runtime_program_;
};

}  // namespace frontend
}  // namespace cinn

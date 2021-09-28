#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <absl/container/flat_hash_map.h>
#include <vector>

#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

/**
 * The executor for a model.
 */
class Interpreter final {
 public:
  Interpreter(const std::vector<std::string>& input_names, const std::vector<hlir::framework::shape_t>& input_shapes);

  /**
   * Load a Paddle model.
   * @param model_dir The directory path to the model.
   * @param params_combined Whether the parameters are composed to a single file.
   */
  void LoadPaddleModel(const std::string& model_dir, const Target& target, bool params_combined = false);

  /**
   * Run the executor.
   */
  void Run();

  hlir::framework::Tensor GetTensor(const std::string& name);

  std::shared_ptr<hlir::framework::Scope> scope();

  ~Interpreter();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace frontend
}  // namespace cinn

#include "cinn/frontend/executor.h"

#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

void Executor::LoadPaddleModel(const std::string& model_dir, bool params_combined) {
  auto [program, var_map, var_map_paddle_to_program] = LoadPaddleProgram(model_dir, scope_.get(), params_combined);
  program_.reset(program.release());
  var_map_                = var_map;
  var_map_paddle_to_cinn_ = var_map_paddle_to_program;

  Build(input_names_, input_shapes_);
}

void Executor::Run() { runtime_program_->Execute(); }

hlir::framework::Tensor Executor::GetTensor(const std::string& name) {
  if (scope_->FindVar(name)) return scope_->GetTensor(name);

  auto it = var_map_paddle_to_cinn_.find(name);
  if (it == var_map_paddle_to_cinn_.end()) {
    LOG(FATAL) << "No variable called [" << name
               << "] found in executor\nThe existing vars: " << utils::Join(scope_->var_names(), ", ");
  }
  return scope_->GetTensor(it->second);
}

void Executor::Build(const std::vector<std::string>& input_names,
                     const std::vector<hlir::framework::shape_t>& input_shapes) {
  CHECK(!input_names.empty());
  CHECK(!var_map_.empty());
  CHECK_EQ(input_names.size(), input_shapes.size());

  std::vector<Variable> input_vars;
  std::transform(input_names.begin(), input_names.end(), std::back_inserter(input_vars), [&](const std::string& x) {
    return var_map_.at(x);
  });

  for (int i = 0; i < input_vars.size(); i++) input_vars[i]->shape = input_shapes[i];

  program_->SetInputs({input_vars});
  program_->Validate();

  LOG(INFO) << "Program:\n" << *program_;

  auto graph = std::make_shared<hlir::framework::Graph>(*program_);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  Target target = common::DefaultHostTarget();
  scope_        = hlir::framework::BuildScope(target, graph, scope_);
  graph_compiler_.reset(new hlir::framework::GraphCompiler(target, scope_, graph));
  runtime_program_ = graph_compiler_->Build();
}

std::shared_ptr<hlir::framework::Scope> Executor::scope() {
  CHECK(scope_);
  return scope_;
}

}  // namespace cinn::frontend

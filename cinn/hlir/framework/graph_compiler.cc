#include "cinn/hlir/framework/graph_compiler.h"

#include <unordered_map>
#include "cinn/hlir/framework/instruction.h"

namespace cinn {
namespace hlir {
namespace framework {
using StrategyFunction = std::function<std::shared_ptr<OpStrategy>(
    const NodeAttr, const std::vector<ir::Tensor>, common::Type, const common::Target)>;

std::unique_ptr<Program> GraphCompiler::Build() {
  auto [nodes, edges] = graph_->topological_order();
  for (auto n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
      m_builder_.AddFunction(lowered_func);
    }
  }

  LOG(INFO) << "Compile the module";
  // compile the module
  CHECK(compiler_);
  compiler_->Build(m_builder_.Build());

  return std::unique_ptr<Program>(new Program(scope_, BuildInstructions()));
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions() {
  std::vector<std::unique_ptr<Instruction>> instructions;

  auto [nodes, edges] = graph_->topological_order();
  for (auto n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), OpGetInputNames(node), OpGetOutputNames(node)));
      auto* fn = compiler_->Lookup(GenOpFuncName(node));
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      instructions.push_back(std::move(instr));
    }
  }
  return instructions;
}

ir::LoweredFunc GraphCompiler::GetOpFunc(const Node* node) {
  auto strategy = Operator::GetAttr<StrategyFunction>("CINNStrategy");
  auto res      = graph_->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infer_shape");
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  for (auto i : node->inlinks()) {
    std::string input_id      = i->source()->as<NodeData>()->id();
    std::vector<int> in_shape = res[input_id];
    lang::Placeholder<float> temp(input_id, in_shape);
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }
  common::Type type;
  auto impl = SelectImpl(strategy[node->op()](node->attrs, inputs, type, target_));

  common::CINNValuePackShared C = impl->fcompute(common::CINNValuePack::Make(cinn_inputs));
  C                             = impl->fschedule(C);
  for (int i = 0; i < C.get()->size(); i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower(node->id(), inputs);
}

std::vector<std::string> GraphCompiler::OpGetInputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto i : node->inlinks()) {
    res.push_back(i->source()->as<NodeData>()->id());
  }
  return res;
}

std::vector<std::string> GraphCompiler::OpGetOutputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto i : node->outlinks()) {
    res.push_back(i->sink()->as<NodeData>()->id());
  }
  return res;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

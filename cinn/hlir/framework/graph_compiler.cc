#include "cinn/hlir/framework/graph_compiler.h"
#include <unordered_map>
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<Program> GraphCompiler::Build() {
  auto [nodes, edges] = graph_->topological_order();
  for (auto& n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
      LOG(INFO) << "CodeGen: " << lowered_func;
      m_builder_.AddFunction(lowered_func);
    }
  }
  // compile the module
  if (!compiler_) {
    compiler_ = backends::Compiler::Create(target_);
  }
  compiler_->Build(m_builder_.Build());

  return std::unique_ptr<Program>(new Program(scope_, BuildInstructions()));
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions() {
  std::vector<std::unique_ptr<Instruction>> instructions;

  auto [nodes, edges] = graph_->topological_order();
  for (auto& n : nodes) {
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
  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetMutableAttrs<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto& dtype_dict = graph_->GetMutableAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  for (auto& i : node->inlinks()) {
    std::string input_id      = i->source()->as<NodeData>()->id();
    std::vector<int> in_shape = shape_dict[input_id];
    Type dtype                = dtype_dict[input_id];
    CHECK_EQ(dtype, Float(32)) << "The dtype of node " << input_id
                               << " is not float! Other dtype is not implemented yet.";
    lang::Placeholder<float> temp(input_id, in_shape);
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }

  std::vector<Type> out_types;
  for (auto& out : node->outlinks()) {
    std::string out_id = out->source()->as<NodeData>()->id();
    Type dtype         = dtype_dict[out_id];
    out_types.push_back(dtype);
  }

  auto impl = OpStrategy::SelectImpl(strategy[node->op()](node->attrs, inputs, out_types, target_));

  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
  C                       = impl->fschedule(C);
  for (int i = 0; i < C.get()->size(); i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower(GenOpFuncName(node), inputs);
  return func;
}

std::vector<std::string> GraphCompiler::OpGetInputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->inlinks()) {
    res.push_back(i->source()->as<NodeData>()->id());
  }
  return res;
}

std::vector<std::string> GraphCompiler::OpGetOutputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->outlinks()) {
    res.push_back(i->sink()->as<NodeData>()->id());
  }
  return res;
}

std::shared_ptr<Scope> BuildScope(Target target, const std::shared_ptr<Graph>& graph) {
  auto shape_dict = graph->GetAttrs<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto dtype_dict = graph->GetAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  auto scope      = std::make_shared<Scope>();
  for (auto& iter : shape_dict) {
    auto* var    = scope->Var<Tensor>(iter.first);
    auto& tensor = std::get<Tensor>(*var);
    std::vector<Shape::dim_t> shape;
    for (auto& shape_dim : iter.second) {
      shape.push_back(Shape::dim_t(shape_dim));
    }
    tensor.Resize(Shape{shape});
    CHECK_EQ(dtype_dict[iter.first], Float(32))
        << "The dtype of node " << iter.first << " is not float! Other dtype is not implemented yet.";
    auto* data = tensor.mutable_data<float>(target);
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

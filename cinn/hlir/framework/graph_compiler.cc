#include "cinn/hlir/framework/graph_compiler.h"

#include <unordered_map>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

void GraphCompiler::PrintFunc() {
  auto [nodes, edges] = graph_->topological_order();
  for (auto& n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
    }
  }
}

std::unique_ptr<Program> GraphCompiler::Build() {
  auto [nodes, edges] = graph_->topological_order();
  for (auto& n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
      m_builder_.AddFunction(lowered_func);
    }
  }
  // compile the module
  if (!compiler_) {
    compiler_ = backends::Compiler::Create(target_);
  }

  if (this->target_.arch == Target::Arch::X86) {
    CodeGenCX86 codegen(this->target_, CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(m_builder_.Build(), CodeGenC::OutputKind::CImpl);
    LOG(INFO) << "[Debug] C Code is:\n" << out;
  } else if (this->target_.arch == Target::Arch::NVGPU) {
    backends::CodeGenCUDA_Dev codegen(this->target_);
    auto out = codegen.Compile(m_builder_.Build());
    LOG(INFO) << "[Debug] CUDA Code is:\n" << out;
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
  auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  std::vector<std::vector<int>> output_shapes;
  VLOG(2) << "GetOpFunc of op " << node->id();
  for (auto& i : node->inlinks_in_order()) {
    std::string input_id = i->source()->as<NodeData>()->id();
    auto in_shape        = shape_dict.at(input_id);
    Type dtype           = dtype_dict.at(input_id);
    CHECK_EQ(dtype, Float(32)) << "The dtype of node " << input_id
                               << " is not float! Other dtype is not implemented yet.";
    lang::Placeholder<float> temp(input_id, in_shape);
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }
  std::vector<Type> out_types;
  for (auto& out : node->outlinks_in_order()) {
    std::string out_id = out->sink()->safe_as<NodeData>()->id();
    auto out_shape     = shape_dict.at(out_id);
    Type dtype         = dtype_dict.at(out_id);
    output_shapes.push_back(out_shape);
    out_types.push_back(dtype);
  }
  auto impl = OpStrategy::SelectImpl(strategy[node->op()](node->attrs, inputs, out_types, output_shapes, target_));

  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
  poly::StageMap stages   = C.back();
  // make sure all the tensors in the stages before schedule launch.
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    stages->InsertLazily(temp.as_tensor_ref());
  }

  C = impl->fschedule(C);
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }

  auto func = Lower(GenOpFuncName(node), stages, inputs);
  VLOG(2) << "The function of node [" << node->attrs.node_name << "] is:\n" << func;
  return func;
}

std::vector<std::string> GraphCompiler::OpGetInputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->inlinks_in_order()) {
    res.push_back(i->source()->as<NodeData>()->id());
  }
  return res;
}

std::vector<std::string> GraphCompiler::OpGetOutputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->outlinks_in_order()) {
    res.push_back(i->sink()->as<NodeData>()->id());
  }
  return res;
}

std::shared_ptr<Scope> BuildScope(Target target, const std::shared_ptr<Graph>& graph, std::shared_ptr<Scope> scope) {
  auto& shape_dict = graph->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph->GetAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  if (!scope) scope = std::make_shared<Scope>();
  for (auto& iter : shape_dict) {
    auto* var    = scope->Var<Tensor>(iter.first);
    auto& tensor = std::get<Tensor>(*var);
    std::vector<Shape::dim_t> shape;
    for (auto& shape_dim : iter.second) {
      shape.push_back(Shape::dim_t(shape_dim));
    }
    VLOG(3) << "Tensor [" << iter.first << "] resize to " << utils::Join(shape, ",");
    tensor->Resize(Shape{shape});
    CHECK_EQ(dtype_dict.at(iter.first), Float(32))
        << "The dtype of node " << iter.first << " is not float! Other dtype is not implemented yet.";
    tensor->mutable_data<float>(target);
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

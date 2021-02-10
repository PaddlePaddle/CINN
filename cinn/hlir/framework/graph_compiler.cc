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

std::string GraphCompiler::GenSourceCode() {
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

  auto build_module = m_builder_.Build();

  return compiler_->GetSourceCode(build_module);
}

std::unique_ptr<Program> GraphCompiler::Build(const std::string& code) {
  auto [nodes, edges] = graph_->topological_order();
  for (int i = 0; i < nodes.size(); i++) {
    auto* node = nodes[i]->safe_as<Node>();
    if (node && node->attrs.attr_store.count("FuseNumber") > 0) {
      int fuse_number = std::get<int>(node->attrs.attr_store["FuseNumber"]);
      std::vector<Node*> fuse_nodes;
      for (int j = 0; j < fuse_number; j++) {
        CHECK_LT(i, nodes.size());
        auto* temp_node = nodes[i]->safe_as<Node>();
        CHECK(temp_node) << "Temp node null Error!!";
        fuse_nodes.push_back(temp_node);
        // Here nodes holds both op node and tensor node. Since a op node is
        // connnected to a tensor node, in order to visit only the op node,
        // we use i = i + 2 instead of i = i + 1.
        i = i + 2;
      }
      // When jump out of the previous loop, we did one more time of i = i + 2.
      // So to ensure each node is traversed, we do i = i - 2.
      i                 = i - 2;
      auto lowered_func = GetOpFunc(fuse_nodes);
      m_builder_.AddFunction(lowered_func);
      continue;
    } else if (node) {
      auto lowered_func = GetOpFunc(node);
      m_builder_.AddFunction(lowered_func);
    }
  }
  // compile the module
  if (!compiler_) {
    compiler_ = backends::Compiler::Create(target_);
  }

  auto build_module = m_builder_.Build();

  if (this->target_.arch == Target::Arch::X86) {
    CodeGenCX86 codegen(this->target_, CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(build_module, CodeGenC::OutputKind::CImpl);
    LOG(INFO) << "[X86] C Code is:\n" << out;
  }

  compiler_->Build(build_module, code);

  return std::unique_ptr<Program>(new Program(scope_, BuildInstructions()));
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions() {
  std::vector<std::unique_ptr<Instruction>> instructions;

  auto [nodes, edges] = graph_->topological_order();
  for (int i = 0; i < nodes.size(); i++) {
    auto* node = nodes[i]->safe_as<Node>();
    if (node && node->attrs.attr_store.count("FuseNumber") > 0) {
      int fuse_number = std::get<int>(node->attrs.attr_store["FuseNumber"]);
      auto* end_node  = nodes[i + 2 * fuse_number - 2]->safe_as<Node>();
      auto instr      = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), OpGetInputNames(node), OpGetOutputNames(end_node)));
      auto* fn = compiler_->Lookup(GenOpFuncName(node) + "_fused");
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      instructions.push_back(std::move(instr));
      i = i + 2 * fuse_number - 2;
      continue;
    } else if (node) {
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

  auto func = Lower(GenOpFuncName(node), stages, inputs, {}, {}, nullptr, this->target_);
  VLOG(2) << "The function of node [" << node->attrs.node_name << "] is:\n" << func;
  return func;
}

ir::LoweredFunc GraphCompiler::GetOpFunc(const std::vector<Node*>& nodes) {
  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  std::vector<ir::Tensor> inputs;
  poly::StageMap stages;
  std::vector<int> init_shape{1};
  int fuse_number = nodes.size();
  lang::Placeholder<float> temp_out_ph("init", init_shape);
  ir::Expr temp_out(temp_out_ph);
  int index = 0;
  for (auto& node : nodes) {
    std::vector<ir::Tensor> temp_inputs;
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<std::vector<int>> output_shapes;
    int input_index = 0;
    for (auto& i : node->inlinks_in_order()) {
      if (index > 0 && input_index == 0) {
        cinn_inputs.push_back(common::CINNValue(temp_out));
        temp_inputs.push_back(temp_out.as_tensor_ref());
      } else {
        std::string input_id = i->source()->as<NodeData>()->id();
        auto in_shape        = shape_dict.at(input_id);
        Type dtype           = dtype_dict.at(input_id);
        CHECK_EQ(dtype, Float(32)) << "The dtype of node " << input_id
                                   << " is not float! Other dtype is not implemented yet.";
        lang::Placeholder<float> temp(input_id, in_shape);
        inputs.push_back(temp);
        temp_inputs.push_back(temp);
        cinn_inputs.push_back(common::CINNValue(temp));
      }
      input_index++;
    }
    std::vector<Type> out_types;
    for (auto& out : node->outlinks_in_order()) {
      std::string out_id = out->sink()->safe_as<NodeData>()->id();
      auto out_shape     = shape_dict.at(out_id);
      Type dtype         = dtype_dict.at(out_id);
      output_shapes.push_back(out_shape);
      out_types.push_back(dtype);
    }
    auto impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, temp_inputs, out_types, output_shapes, target_));

    common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
    C                       = impl->fschedule(C);
    CHECK_GE(C.size(), 2);
    ir::Expr temp0             = C[0];
    temp_out                   = temp0;
    poly::StageMap temp_stages = C.back();

    for (auto& i : temp_stages) {
      stages->InsertLazily(ir::Tensor(i.second->tensor()), i.second.get());
    }

    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      stages->InsertLazily(temp.as_tensor_ref(), temp_stages[temp.as_tensor_ref()]);
      if (index < fuse_number - 1 && !temp.as_tensor_ref()->is_reduce_tensor()) {
        stages[temp.as_tensor_ref()]->ComputeInline();
      } else {
        inputs.push_back(temp.as_tensor_ref());
      }
    }
    index++;
  }

  auto func = Lower(GenOpFuncName(nodes[0]) + "_fused", stages, inputs, {}, {}, nullptr, this->target_);
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

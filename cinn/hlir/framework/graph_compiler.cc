#include "cinn/hlir/framework/graph_compiler.h"

#include <unordered_map>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace framework {
// Store params from node to instruction
void AddAttrs(const std::unordered_map<std::string, AttrType>& attrs_store,
              const std::vector<std::string>& attrs_name,
              Instruction* instr) {
  for (auto& attr : attrs_name) {
    if (attrs_store.find(attr) != attrs_store.end()) {
      switch (attrs_store.at(attr).index()) {
        case 2:
          instr->attrs.push_back(std::get<int>(attrs_store.at(attr)));
          break;
        case 3:
          instr->str_attrs.push_back(std::get<std::string>(attrs_store.at(attr)));
          break;
        case 5:
          auto temp = std::get<std::vector<int>>(attrs_store.at(attr));
          instr->attrs.insert(instr->attrs.end(), temp.begin(), temp.end());
          break;
      }
    } else {
      LOG(ERROR) << "Param " << attr << " missed! Please check.";
    }
  }
}

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
      std::vector<std::string> inputNames;
      std::vector<std::string> outputNames;
      for (int j = 0; j < fuse_number; j++) {
        auto* temp_node = nodes[i + 2 * j]->safe_as<Node>();
        CHECK(temp_node);
        auto temp_inputnames = OpGetInputNames(temp_node);
        if (j == 0) {
          inputNames.insert(inputNames.end(), temp_inputnames.begin(), temp_inputnames.end());
        } else {
          inputNames.insert(inputNames.end(), temp_inputnames.begin() + 1, temp_inputnames.end());
        }
        if (j == fuse_number - 1) {
          auto temp_outputnames = OpGetOutputNames(temp_node);
          outputNames.insert(outputNames.end(), temp_outputnames.begin(), temp_outputnames.end());
        }
      }
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), inputNames, outputNames, node->op()->name + "_fused"));
      auto* fn = compiler_->Lookup(GenOpFuncName(node) + "_fused");
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      instructions.push_back(std::move(instr));
      i = i + 2 * fuse_number - 2;
      continue;
    } else if (node) {
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), OpGetInputNames(node), OpGetOutputNames(node), node->op()->name));
      if (target_.arch == Target::Arch::NVGPU) {
        if (node->op()->name == "conv2d") {
          auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          AddAttrs(node->attrs.attr_store, {"padding", "stride", "dilation"}, instr.get());
          if (node->attrs.attr_store.find("groups") != node->attrs.attr_store.end()) {
            auto groups = std::get<int>(node->attrs.attr_store.at("groups"));
            instr->attrs.push_back(groups);
          } else {
            instr->attrs.push_back(1);
          }
          CHECK(!node->outlinks_in_order().empty());
          auto& out_node = node->outlinks_in_order().front();
          std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
          auto out_shape     = shape_dict.at(out_id);
          instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          CHECK_EQ(instr->attrs.size(), 19UL);
        } else if (node->op()->name == "depthwise_conv2d") {
          auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          AddAttrs(node->attrs.attr_store, {"padding", "stride", "dilation"}, instr.get());
          if (node->attrs.attr_store.find("groups") != node->attrs.attr_store.end()) {
            auto groups = std::get<int>(node->attrs.attr_store.at("groups"));
            instr->attrs.push_back(groups);
          } else {
            instr->attrs.push_back(instr->attrs[1]);
          }
          for (auto& out_node : node->outlinks_in_order()) {
            std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
            auto out_shape     = shape_dict.at(out_id);
            instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          }
          CHECK_EQ(instr->attrs.size(), 19UL);
        } else if (node->op()->name == "pool2d") {
          auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            CHECK_EQ(in_shape.size(), 4UL);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          bool global_pooling = false;
          if (node->attrs.attr_store.find("global_pooling") != node->attrs.attr_store.end()) {
            global_pooling = std::get<bool>(node->attrs.attr_store.at("global_pooling"));
          }
          if (node->attrs.attr_store.find("kernel_size") != node->attrs.attr_store.end()) {
            if (global_pooling == false) {
              auto padding = std::get<std::vector<int>>(node->attrs.attr_store.at("kernel_size"));
              instr->attrs.insert(instr->attrs.end(), padding.begin(), padding.end());
            } else {
              instr->attrs.push_back(instr->attrs[2]);
              instr->attrs.push_back(instr->attrs[3]);
            }
          }
          if (node->attrs.attr_store.find("padding_size") != node->attrs.attr_store.end()) {
            if (global_pooling == false) {
              auto stride = std::get<std::vector<int>>(node->attrs.attr_store.at("padding_size"));
              instr->attrs.insert(instr->attrs.end(), stride.begin(), stride.end());
            } else {
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
            }
          }
          AddAttrs(node->attrs.attr_store, {"stride_size", "pool_type"}, instr.get());

          for (auto& out_node : node->outlinks_in_order()) {
            std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
            auto out_shape     = shape_dict.at(out_id);
            instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          }
          CHECK_EQ(instr->attrs.size(), 16UL);
          CHECK_EQ(instr->str_attrs.size(), 1UL);
        } else if (node->op()->name == "softmax") {
          auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          AddAttrs(node->attrs.attr_store, {"axis"}, instr.get());
        } else if (node->op()->name == "mul") {
          auto& shape_dict = graph_->GetAttrs<std::unordered_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          if (node->attrs.attr_store.find("x_num_col_dims") != node->attrs.attr_store.end()) {
            auto axis = std::get<int>(node->attrs.attr_store.at("x_num_col_dims"));
            instr->attrs.push_back(axis);
          } else {
            instr->attrs.push_back(1);
          }
          if (node->attrs.attr_store.find("y_num_col_dims") != node->attrs.attr_store.end()) {
            auto axis = std::get<int>(node->attrs.attr_store.at("y_num_col_dims"));
            instr->attrs.push_back(axis);
          } else {
            instr->attrs.push_back(1);
          }
        }
      }
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
  VLOG(2) << "GetOpFunc of fused op " << nodes[0]->id();
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
      } else if (index < fuse_number - 1 && temp.as_tensor_ref()->is_reduce_tensor()) {
        temp.as_tensor_ref()->WithBuffer("local", "_" + temp.as_tensor_ref()->name + "_temp_buffer");
        stages[temp.as_tensor_ref()]->SetScope(poly::ScopeKind::kLocal);
      } else {
        inputs.push_back(temp.as_tensor_ref());
      }
    }
    index++;
  }

  for (auto& s : stages) {
    if (s.second->tensor()->is_reduce_tensor()) {
      stages[inputs.back()]->CopyTransform(s.second.get());
      stages[inputs.back()]->CopyLoopInfo(s.second->forloop_infos(), s.second->transform());
    }
  }
  auto func = Lower(GenOpFuncName(nodes[0]) + "_fused", stages, inputs, {}, {}, nullptr, this->target_);
  VLOG(3) << "The function of fused node [" << func->name << "] is:\n" << func;
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

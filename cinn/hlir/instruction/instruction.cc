#include "cinn/hlir/instruction/instruction.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/ir/ir_base.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace instruction {

void InsToTensor(Instruction* ins, std::map<Instruction*, ir::Tensor>& mapper) {
  if (ins->GetType() == "input") {
    std::vector<Expr> shape;
    for (auto i : ins->GetShape()) {
      shape.push_back(ir::Expr(i));
    }
    Placeholder<float> temp(ins->GetName(), shape);
    mapper[ins] = temp.tensor();
  } else if (ins->GetType() == "Pad") {
    CHECK_EQ(1, ins->GetInlinks().size());
    auto i = *(ins->GetInlinks().begin());
    CHECK(mapper.count(i) > 0) << "Wrong order of " << i->GetType() << ", " << ins->GetType();
    std::vector<ir::Expr> param;
    CHECK(ins->GetAttrs().count("pad_param") > 0);
    for (auto j : (ins->GetAttrs())["pad_param"]) {
      param.push_back(ir::Expr(j));
    }
    auto temp   = pe::Pad(mapper[i], param);
    mapper[ins] = temp;
  } else if (ins->GetType() == "ConvBroadcast") {
    CHECK_EQ(2, ins->GetInlinks().size());
    auto it = ins->GetInlinks().begin();
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input0 = mapper[(*it)];
    it++;
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input1 = mapper[(*it)];
    CHECK(ins->GetAttrs().count("conv_param") > 0);
    auto vec_param = ins->GetAttrs()["conv_param"];
    CHECK_EQ(vec_param.size(), 4);
    auto temp = pe::ConvBroadcast(
        input0, input1, ir::Expr(vec_param[0]), ir::Expr(vec_param[1]), ir::Expr(vec_param[2]), ir::Expr(vec_param[3]));
    mapper[ins] = temp;
  } else if (ins->GetType() == "ReduceSum") {
    CHECK_EQ(1, ins->GetInlinks().size());
    auto it = ins->GetInlinks().begin();
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input0 = mapper[(*it)];
    CHECK(ins->GetAttrs().count("reduce_param") > 0);
    auto vec_param = ins->GetAttrs()["reduce_param"];
    auto temp      = pe::ReduceSum(input0, vec_param);
    mapper[ins]    = temp;
  } else if (ins->GetType() == "Add") {
    CHECK_EQ(2, ins->GetInlinks().size());
    auto it = ins->GetInlinks().begin();
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input0 = mapper[(*it)];
    it++;
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input1 = mapper[(*it)];
    auto temp         = pe::Add(input0, input1);
    mapper[ins]       = temp;
  } else if (ins->GetType() == "Relu") {
    CHECK_EQ(1, ins->GetInlinks().size());
    auto it = ins->GetInlinks().begin();
    CHECK(mapper.count((*it)) > 0);
    ir::Tensor input0 = mapper[(*it)];
    auto temp         = pe::Relu(input0, 0.f);
    mapper[ins]       = temp;
  }
}

Instruction* CreateInput(const std::vector<int>& shape, const std::string& name) {
  Instruction* res(new Instruction("input", shape, name));
  return res;
}

Instruction* Pad(Instruction* arg0, const std::vector<int>& param) {
  Instruction* res(new Instruction("Pad"));
  res->inlinks_.insert(arg0);
  res->attrs_["pad_param"] = param;
  return res;
}

Instruction* ConvBroadcast(Instruction* arg0, Instruction* arg1, const std::vector<int>& param) {
  Instruction* res(new Instruction("ConvBroadcast"));
  res->inlinks_.insert(arg0);
  res->inlinks_.insert(arg1);
  res->attrs_["conv_param"] = param;
  return res;
}

Instruction* ReduceSum(Instruction* arg0, const std::vector<int>& param) {
  Instruction* res(new Instruction("ReduceSum"));
  res->inlinks_.insert(arg0);
  res->attrs_["reduce_param"] = param;
  return res;
}

Instruction* Add(Instruction* arg0, Instruction* arg1) {
  Instruction* res(new Instruction("Add"));
  res->inlinks_.insert(arg0);
  res->inlinks_.insert(arg1);
  return res;
}

Instruction* Relu(Instruction* arg0) {
  Instruction* res(new Instruction("Relu"));
  res->inlinks_.insert(arg0);
  return res;
}

std::string Instruction::GetFunction() {
  std::set<Instruction*> all_ins;
  std::vector<Instruction*> queue;
  queue.push_back(this);
  all_ins.insert(this);
  int i = 0;
  while (i < queue.size()) {
    for (auto* j : queue[i]->GetInlinks()) {
      if (all_ins.count(j) == 0) {
        all_ins.insert(j);
        queue.push_back(j);
      }
    }
    i++;
  }
  std::vector<Instruction*> toposort;
  while (!all_ins.empty()) {
    for (auto it = all_ins.begin(); it != all_ins.end();) {
      int in_degree = 0;
      for (auto* j : (*it)->GetInlinks()) {
        if (all_ins.count(j) > 0) in_degree++;
      }
      if (in_degree == 0) {
        toposort.push_back((*it));
        all_ins.erase(it++);
      } else {
        it++;
      }
    }
  }
  std::map<Instruction*, ir::Tensor> mapper;
  for (auto& i : toposort) {
    InsToTensor(i, mapper);
  }

  std::vector<ir::Tensor> tensor_args   = {mapper[this]};
  std::vector<ir::Tensor> other_tensors = {};
  for (auto& i : mapper) {
    if (i.second->is_compute_node() && i.first != this)
      other_tensors.push_back(i.second);
    else if (!i.second->is_compute_node())
      tensor_args.push_back(i.second);
  }

  auto stages = pe::PrepareStages(tensor_args, other_tensors);

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module01", target);
  auto func = Lower("fn", stages, tensor_args);
  builder.AddFunction(func);
  std::string res = utils::GetStreamCnt(func);
  return res;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

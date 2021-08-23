#pragma once
#include <glog/logging.h>

#include <any>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/common/type.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/reduction.h"
#include "cinn/hlir/pe/schedule.h"

namespace cinn {
namespace hlir {
namespace instruction {

/**
 * The Instruction is like a finer-grained op primitive. Users can use it to construct different computations(as a new
 * operator). And the schedule of it can be generated automatically based on rules.
 */
class Instruction {
 public:
  std::string GetFunction();
  const std::string GetType() const { return type_; }
  const std::string GetName() const { return name_; }
  std::vector<int> GetShape() { return shape_; }
  std::set<Instruction*>& GetInlinks() { return inlinks_; }
  std::map<std::string, std::vector<int>>& GetAttrs() { return attrs_; }
  Instruction(std::string type, std::vector<int> shape = {}, std::string name = "")
      : type_(type), shape_(shape), name_(name) {}
  std::set<Instruction*> inlinks_;
  const std::string type_;
  std::map<std::string, std::vector<int>> attrs_;
  std::vector<int> shape_;
  const std::string name_;
};

Instruction* CreateInput(const std::vector<int>& shape, const std::string& name);

Instruction* Pad(Instruction* arg0, const std::vector<int>& param);

Instruction* ConvBroadcast(Instruction* arg0, Instruction* arg1, const std::vector<int>& param);

Instruction* ReduceSum(Instruction* arg0, const std::vector<int>& param);

Instruction* Add(Instruction* arg0, Instruction* arg1);

Instruction* Relu(Instruction* arg0);

ir::Tensor InsToTensor(Instruction* ins);

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

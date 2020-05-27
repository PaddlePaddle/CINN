#pragma once
#include <string>
#include <vector>

#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace hlir {
namespace instruction {

class ParameterInstruction : public Instruction {
 public:
  ParameterInstruction(int param_offset, const std::string& name, const Shape& shape)
      : name_(name), Instruction(InstrCode::Parameter, shape), param_offset_(param_offset) {}

  std::string to_debug_string() override;

  const std::string& name() const { return name_; }

  std::string id() const override;

  int param_offset() const { return param_offset_; }

 private:
  std::string name_;
  int param_offset_{-1};
};

class CompareInstruction : public Instruction {
 public:
  CompareInstruction(const Shape& shape, Instruction* arg0, Instruction* arg1, CompareDirection direction)
      : Instruction(InstrCode::Compare, shape), direction_(direction) {
    AppendOperand(arg0);
    AppendOperand(arg1);
  }

 private:
  CompareDirection direction_;
};

class ReduceInstruction : public Instruction {
 public:
  ReduceInstruction(const Shape& shape,
                    Instruction* arg0,
                    Instruction* init_value,
                    const std::vector<int>& reduce_dimensions,
                    Computation* reduce_computation)
      : Instruction(InstrCode::Reduce, shape),
        init_value_(init_value),
        reduce_dimensions_(reduce_dimensions),
        reduce_computation_(reduce_computation) {
    AppendOperand(arg0);
    AppendOperand(init_value);
  }

 private:
  Instruction* init_value_{};
  std::vector<int> reduce_dimensions_;
  Computation* reduce_computation_{};
};

class BroadcastInstruction : public Instruction {
 public:
  BroadcastInstruction(const Shape& shape, const std::vector<int>& dimensions)
      : Instruction(InstrCode::Broadcast, shape), dimensions_(dimensions) {}

 private:
  std::vector<int> dimensions_;
};

class TransposeInstruction : public Instruction {
 public:
  TransposeInstruction(const Shape& shape, const std::vector<int>& dimensions)
      : Instruction(InstrCode::Transpose, shape), dimensions_(dimensions) {}

 private:
  std::vector<int> dimensions_;
};

class ConstantInstruction : public Instruction {
 public:
  ConstantInstruction(const Shape& shape, const std::vector<char>& data)
      : Instruction(InstrCode::Constant, shape), data_(data) {}

 private:
  std::vector<char> data_;
};

class CallInstruction : public Instruction {
 public:
  CallInstruction(const Computation* computation,
                  const std::vector<Instruction*>& args,
                  const std::vector<Shape>& shapes,
                  const std::vector<std::string>& tensor_names,
                  const std::vector<cinn::common::Type>& types)
      : Instruction(InstrCode::Call, Shape{}), computation_(computation) {
    for (auto* arg : args) {
      AppendOperand(arg);
    }

    CHECK_EQ(tensor_names.size(), types.size());
    CHECK_EQ(shapes.size(), tensor_names.size());

    ret_tensor_names_ = tensor_names;
    ret_types_        = types;
    ret_shapes_       = shapes;
  }

  const std::vector<std::string>& ret_tensor_names() const { return ret_tensor_names_; }
  const std::vector<cinn::common::Type>& ret_types() const { return ret_types_; }
  const std::vector<Shape>& ret_shapes() const { return ret_shapes_; }

  bool can_inlined() const { return ret_tensor_names().size() == 1UL; }

  const Computation* computation() const { return computation_; }

  std::string to_debug_string() override;

 private:
  std::vector<std::string> ret_tensor_names_;
  std::vector<cinn::common::Type> ret_types_;
  std::vector<Shape> ret_shapes_;

  const Computation* computation_;
};

class CustomCallInstruction : public Instruction {
 public:
  CustomCallInstruction(const Shape& shape,
                        const std::vector<Instruction*>& args,
                        const std::string& call_target,
                        const std::string& tag)
      : Instruction(InstrCode::CustomCall, shape), call_target_(call_target), args_(args) {}

 private:
  std::string call_target_;
  std::vector<Instruction*> args_;
};

/**
 * A tuple as the return values of a Call.
 */
class Tuple : public Instruction {
 public:
  explicit Tuple(Instruction* call) : Instruction(InstrCode::Tuple, Shape{}), call_(call) { operands_.push_back(call); }
  explicit Tuple(const std::vector<Instruction*>& items)
      : Instruction(InstrCode::Tuple, Shape{}), call_(nullptr), items_(items) {
    CHECK(!items.empty());
    operands_.insert(operands_.end(), items.begin(), items.end());
  }

  std::unique_ptr<Instruction> Get(int i);

  const Instruction* call() const { return call_; }
  const std::vector<Instruction*>& items() const { return items_; }
  std::string to_debug_string() override;

 private:
  const Instruction* call_;
  std::vector<Instruction*> items_;
};

class TupleGet : public Instruction {
 public:
  explicit TupleGet(Instruction* tuple, int offset)
      : Instruction(InstrCode::TupleGet, Shape{}), tuple_(tuple), offset_(offset) {
    CHECK_LE(offset, 0);

    auto* tuple_node = tuple->As<Tuple>();
    if (tuple_node->call()) {
      auto* call_node = tuple_node->call()->As<CallInstruction>();
      shape_          = call_node->ret_shapes()[offset];
      type_           = call_node->ret_types()[offset];
    } else if (!tuple_node->items().empty()) {
      shape_ = tuple_node->items()[offset]->shape();
      type_  = tuple_node->items()[offset]->type();
    } else {
      NOT_IMPLEMENTED
    }

    operands_.push_back(tuple);
  }

  const Tuple* tuple() const { return tuple_->As<Tuple>(); }
  int offset() const { return offset_; }

 private:
  const Instruction* tuple_{};
  int offset_{-1};
};

class Conv : public Instruction {
 public:
  Conv(Instruction* I, Instruction* W, int pad_h, int pad_w, int stride_h, int stride_w)
      : Instruction(InstrCode::Conv, Shape()), pad_h_(pad_h), pad_w_(pad_w), stride_h_(stride_h), stride_w_(stride_w) {
    AppendOperand(I);
    AppendOperand(W);
  }

  int pad_h() const { return pad_h_; }
  int pad_w() const { return pad_w_; }
  int stride_h() const { return stride_h_; }
  int stride_w() const { return stride_w_; }

 private:
  int pad_h_{};
  int pad_w_{};
  int stride_h_{};
  int stride_w_{};
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

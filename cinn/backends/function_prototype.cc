#include "cinn/backends/function_prototype.h"

#include "cinn/lang/tensor.h"

namespace cinn {
namespace backends {

bool FunctionProto::Match(const ir::Call *op) const {
  if (name != op->name) return false;
  if (ret_type != op->type()) return false;
  if (op->read_args.size() != readonly_arg_types.size()) return false;
  if (op->write_args.size() != mutable_arg_types.size()) return false;

  for (int i = 0; i < op->read_args.size(); i++) {
    if (op->read_args[i].type() != readonly_arg_types[i]) return false;
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    if (op->write_args[i].type() != mutable_arg_types[i]) return false;
  }
  return true;
}

void FunctionProto::AssertMatch(const ir::Call *op) const {
  CHECK_EQ(name, op->name);
  CHECK_EQ(ret_type, op->type()) << "function proto " << name << " check failed";
  CHECK_EQ(op->read_args.size(), readonly_arg_types.size()) << "function proto " << name << " check failed";
  CHECK_EQ(op->write_args.size(), mutable_arg_types.size()) << "function proto " << name << " check failed";

  auto get_type = [](Expr u) {
    if (u.as_tensor() || u.as_buffer()) {
      Type t = u.type();
      return t.set_cpp_handle();
    }
    return u.type();
  };
  for (int i = 0; i < op->read_args.size(); i++) {
    if (readonly_arg_types[i] == type_of<cinn_buffer_t *>()) {
      if (!op->read_args[i].as_tensor()) continue;
    } else {
      CHECK_EQ(get_type(op->read_args[i]), readonly_arg_types[i]);
    }
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    if (mutable_arg_types[i] == type_of<cinn_buffer_t *>()) {
      if (!op->write_args[i].as_tensor()) continue;
    } else {
      CHECK_EQ(get_type(op->write_args[i]), mutable_arg_types[i]);
    }
  }
}

void FunctionProto::CheckValid() {
  if (ret_type.is_void()) {
    CHECK(!mutable_arg_types.empty())
        << "A void function should have at least one mutable argument to output something";
  } else {
    CHECK(mutable_arg_types.empty()) << "A function with return should not have mutable argument";
  }
}

FunctionProto::shape_inference_t FunctionProto::ShapeFollowNthArgument(int n) {
  return [=](const std::vector<Expr> &args, int value_offset) {
    CHECK_LT(n, args.size());
    auto x = args[n].as_tensor();
    CHECK(x);
    return x->shape;
  };
}

FunctionProto::FunctionProto(const std::string &name,
                             const std::vector<Type> &readonly_arg_types,
                             const std::vector<Type> &mutable_arg_types,
                             Type ret_type,
                             FunctionProto::shape_inference_t shape_inference)
    : name(name),
      readonly_arg_types(readonly_arg_types),
      mutable_arg_types(mutable_arg_types),
      ret_type(ret_type),
      shape_inference(shape_inference) {
  CheckValid();
}

FunctionProto *FunctionProtoRegistry::Lookup(const std::string &name) {
  auto it = data_.find(name);
  if (it != data_.end()) {
    return it->second.get();
  }
  return nullptr;
}

FunctionProto *FunctionProtoRegistry::Register(std::string_view name, FunctionProto *x) {
  VLOG(1) << "Register function prototype "
          << "[" << name << "]";
  data_.emplace(name, std::unique_ptr<FunctionProto>(x));
  return x;
}

std::string FunctionProtoRegistry::debug_string() const {
  std::stringstream ss;
  for (auto &item : data_) {
    ss << item.first << "\n";
  }
  return ss.str();
}
}  // namespace backends
}  // namespace cinn

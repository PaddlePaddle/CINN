#include "cinn/common/ir_util.h"
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"
#include "cinn/utils/string.h"

namespace py = pybind11;

namespace cinn::pybind {

using common::CINNValue;
using common::Object;
using common::Target;
using common::Type;
using utils::GetStreamCnt;
using utils::StringFormat;

namespace {
void BindTarget(py::module *);
void BindType(py::module *);
void BindObject(py::module *);
void BindShared(py::module *);
void BindCinnValue(py::module *);

void BindTarget(py::module *m) {
  py::class_<Target> target(*m, "Target");
  target.def_readwrite("os", &Target::os)
      .def_readwrite("arch", &Target::arch)
      .def_readwrite("bits", &Target::bits)
      .def_readwrite("features", &Target::features)
      .def(py::init<>())
      .def(py::init<Target::OS, Target::Arch, Target::Bit, const std::vector<Target::Feature> &>())
      .def("defined", &Target::defined)
      .def("runtime_arch", &Target::runtime_arch);

  m->def("DefaultHostTarget", &common::DefaultHostTarget).def("DefaultNVGPUTarget", &common::DefaultNVGPUTarget);

  py::enum_<Target::OS> os(target, "OS");
  os.value("Unk", Target::OS::Unk).value("Linux", Target::OS::Linux).value("Windows", Target::OS::Windows);

  py::enum_<Target::Arch> arch(target, "Arch");
  arch.value("Unk", Target::Arch::Unk)
      .value("X86", Target::Arch::X86)
      .value("ARM", Target::Arch::ARM)
      .value("NVGPU", Target::Arch::NVGPU);

  py::enum_<Target::Bit> bit(target, "Bit");
  bit.value("Unk", Target::Bit::Unk).value("k32", Target::Bit::k32).value("k64", Target::Bit::k64);

  py::enum_<Target::Feature> feature(target, "Feature");
  feature.value("JIT", Target::Feature::JIT).value("Debug", Target::Feature::Debug);
}

void BindType(py::module *m) {
  py::class_<Type> type(*m, "Type");
  type.def(py::init<>()).def(py::init<Type::type_t, int, int>());
#define DEFINE_TYPE_METHOD(__name) (type = type.def(#__name, &Type::__name))
  DEFINE_TYPE_METHOD(is_primitive);
  DEFINE_TYPE_METHOD(is_unk);
  DEFINE_TYPE_METHOD(is_void);
  DEFINE_TYPE_METHOD(is_bool);
  DEFINE_TYPE_METHOD(is_vector);
  DEFINE_TYPE_METHOD(is_scalar);
  DEFINE_TYPE_METHOD(is_float);
  DEFINE_TYPE_METHOD(is_int);
  DEFINE_TYPE_METHOD(is_uint);
  DEFINE_TYPE_METHOD(is_string);
  DEFINE_TYPE_METHOD(set_cpp_handle);
  DEFINE_TYPE_METHOD(is_cpp_handle);
  DEFINE_TYPE_METHOD(set_cpp_handle_handle);
  DEFINE_TYPE_METHOD(is_cpp_handle_handle);
  DEFINE_TYPE_METHOD(set_cpp_const);
  DEFINE_TYPE_METHOD(is_cpp_const);
  DEFINE_TYPE_METHOD(set_customized_type);
  DEFINE_TYPE_METHOD(customized_type);
  DEFINE_TYPE_METHOD(is_customized_type);
  DEFINE_TYPE_METHOD(with_bits);
  DEFINE_TYPE_METHOD(with_type);
  DEFINE_TYPE_METHOD(with_cpp_const);
#undef DEFINE_TYPE_METHOD
  type.def("vector_of", &Type::VectorOf)
      .def("element_of", &Type::ElementOf)
      .def("pointer_of", &Type::PointerOf)
      .def("__str__", [](const Type &self) { return GetStreamCnt(self); })
      .def("__repr__", [](const Type &self) { return StringFormat("<Type: %s>", GetStreamCnt(self).c_str()); });

  py::enum_<Type::type_t> type_t(type, "type_t");
  type_t.value("unk", Type::type_t::Unk)
      .value("int", Type::type_t::Int)
      .value("uInt", Type::type_t::UInt)
      .value("float", Type::type_t::Float)
      .value("string", Type::type_t::String)
      .value("void", Type::type_t::Void)
      .value("customized", Type::type_t::Customized)
      .export_values();

  py::enum_<Type::cpp_type_t> cpp_type_t(type, "cpp_type_t");
  cpp_type_t.value("None", Type::cpp_type_t::None)
      .value("Const", Type::cpp_type_t::Const)
      .value("Handle", Type::cpp_type_t::Handle)
      .value("HandleHandle", Type::cpp_type_t::HandleHandle)
      .export_values();

  m->def("Void", &common::Void)
      .def("Int", &common::Int, py::arg("bits"), py::arg("lanes") = 1)
      .def("UInt", &common::UInt, py::arg("bits"), py::arg("lanes") = 1)
      .def("Float", &common::Float, py::arg("bits"), py::arg("lanes") = 1)
      .def("Bool", &common::Bool, py::arg("lanes") = 1)
      .def("String", &common::String);

  m->def(
       "make_const",
       [](const Type &type, int32_t val) -> Expr { return common::make_const(type, val); },
       py::arg("type"),
       py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, int64_t val) -> Expr { return common::make_const(type, val); },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, float val) -> Expr { return common::make_const(type, val); },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, double val) -> Expr { return common::make_const(type, val); },
          py::arg("type"),
          py::arg("val"))
      .def(
          "make_const",
          [](const Type &type, bool val) -> Expr { return common::make_const(type, val); },
          py::arg("type"),
          py::arg("val"));

  m->def("type_of", [](std::string_view dtype) {
    if (dtype == "float32") return common::type_of<float>();
    if (dtype == "float64") return common::type_of<double>();
    if (dtype == "uchar") return common::type_of<unsigned char>();
    if (dtype == "int16") return common::type_of<int16_t>();
    if (dtype == "uint32") return common::type_of<uint32_t>();
    if (dtype == "bool") return common::type_of<bool>();
    if (dtype == "char") return common::type_of<char>();
    if (dtype == "int32") return common::type_of<int32_t>();
    if (dtype == "void") return common::type_of<void>();
    if (dtype == "int8_p") return common::type_of<int8_t *>();
    if (dtype == "void_p") return common::type_of<void *>();
    if (dtype == "void_p_p") return common::type_of<void **>();
    if (dtype == "float32_p") return common::type_of<float *>();
    if (dtype == "float64_p") return common::type_of<double *>();
    if (dtype == "cinn_buffer") return common::type_of<cinn_buffer_t>();
    if (dtype == "cinn_buffer_p") return common::type_of<cinn_buffer_t *>();
    if (dtype == "const_cinn_buffer_p") return common::type_of<const cinn_buffer_t *>();
    if (dtype == "cinn_pod_value") return common::type_of<cinn_pod_value_t>();
    if (dtype == "cinn_pod_value_p") return common::type_of<cinn_pod_value_t *>();

    CINN_NOT_IMPLEMENTED;
    return Void();
  });
}

void BindObject(py::module *m) {
  py::class_<Object, ObjectWrapper> object(*m, "Object");
  object.def("type_info", &Object::type_info);
}

void BindShared(py::module *m) {
  py::class_<common::RefCount> ref_count(*m, "RefCount");
  ref_count.def(py::init<>())
      .def("inc", &common::RefCount::Inc)
      .def("dec", &common::RefCount::Dec)
      .def("is_zero", &common::RefCount::is_zero)
      .def("to_string", &common::RefCount::to_string)
      .def("val", &common::RefCount::val);
}

void BindCinnValue(py::module *m) {
  using common::_CINNValuePack_;
  using common::CINNValuePack;

  DefineShared<_CINNValuePack_>(m, "_CINNValuePack_");

  py::class_<_CINNValuePack_> cinn_value_pack(*m, "_CINNValuePack_");
  cinn_value_pack.def_static("make", &_CINNValuePack_::Make)
      .def("__getitem__", [](_CINNValuePack_ &self, int offset) { return self[offset]; })
      .def("__setitem__", [](_CINNValuePack_ &self, int offset, CINNValue &v) { self[offset] = v; })
      .def("add_value", &_CINNValuePack_::AddValue)
      .def("clear", &_CINNValuePack_::Clear)
      .def("size", &_CINNValuePack_::size)
      .def("__len__", &_CINNValuePack_::size)
      .def("type_info", &_CINNValuePack_::type_info);

  py::class_<CINNValuePack, common::Shared<_CINNValuePack_>> cinn_value_pack_shared(*m, "CINNValuePack");
  cinn_value_pack_shared.def(py::init<_CINNValuePack_ *>())
      .def("__getitem__", [](CINNValuePack &self, int offset) { return self[offset]; })
      .def("__setitem__", [](CINNValuePack &self, int offset, CINNValue &v) { self[offset] = v; });

  py::class_<CINNValue, cinn_pod_value_t> cinn_value(*m, "CINNValue");
  cinn_value.def(py::init<>())
      .def(py::init<cinn_value_t, int>())
      .def(py::init<int32_t>())
      .def(py::init<int64_t>())
      .def(py::init<float>())
      .def(py::init<double>())
      .def(py::init<char *>())
      .def(py::init<cinn_buffer_t *>())
      .def(py::init<void *>())
      .def(py::init<const char *>())
      .def(py::init<const ir::Var &>())
      .def(py::init<const ir::Expr &>())
      .def(py::init<const CINNValuePack &>())
      .def("defined", &CINNValue::defined)
      .def("to_double", [](CINNValue &self) { return static_cast<double>(self); })
      .def("to_float", [](CINNValue &self) { return static_cast<float>(self); })
      .def("to_int32", [](CINNValue &self) { return static_cast<int32_t>(self); })
      .def("to_int64", [](CINNValue &self) { return static_cast<int64_t>(self); })
      .def("to_void_p", [](CINNValue &self) { return static_cast<void *>(self); })
      .def("to_cinn_buffer_p", [](CINNValue &self) { return static_cast<cinn_buffer_t *>(self); })
      .def("to_str", [](CINNValue &self) { return static_cast<char *>(self); })
      .def("to_var", [](CINNValue &self) { return ir::Var(self); })
      .def("to_expr", [](CINNValue &self) { return ir::Expr(self); })
      .def("set", &CINNValue::Set<int32_t>)
      .def("set", &CINNValue::Set<int64_t>)
      .def("set", &CINNValue::Set<float>)
      .def("set", &CINNValue::Set<double>)
      .def("set", &CINNValue::Set<char *>)
      .def("set", &CINNValue::Set<const ir::Var &>)
      .def("set", &CINNValue::Set<const ir::Expr &>)
      .def("set", &CINNValue::Set<cinn_buffer_t *>)
      .def("set", &CINNValue::Set<const CINNValuePack &>)
      .def("set", &CINNValue::Set<const char *>)
      .def("set", &CINNValue::Set<const CINNValue &>);

  auto binary_op_visitor = [](CINNValue &v, auto lhs, auto rhs, auto fn) {
    using lhs_t = decltype(lhs);
    using rhs_t = decltype(rhs);
    if constexpr (std::is_same_v<lhs_t, std::nullptr_t> || std::is_same_v<rhs_t, std::nullptr_t> ||
                  !std::is_same_v<lhs_t, rhs_t>) {
      v = CINNValue();
    } else {
      v.Set(fn(lhs, rhs));
    }
  };

#define DEFINE_BINARY_OP(__op, __fn)                                                         \
  auto __op##_fn = [&](auto x, auto y) {                                                     \
    constexpr auto is_var_x = std::is_same_v<std::decay_t<decltype(x)>, ir::Var>;            \
    constexpr auto is_var_y = std::is_same_v<std::decay_t<decltype(y)>, ir::Var>;            \
    if constexpr (is_var_x && is_var_y) {                                                    \
      return __fn(ir::Expr(x), ir::Expr(y)).as_var_ref();                                    \
    } else {                                                                                 \
      return __fn(x, y);                                                                     \
    }                                                                                        \
  };                                                                                         \
  cinn_value.def(#__op, [&](CINNValue &self, CINNValue &other) {                             \
    auto visitor = [&](auto x, auto y) { return binary_op_visitor(self, x, y, __op##_fn); }; \
    std::visit(visitor, ConvertToVar(self), ConvertToVar(other));                            \
    return self;                                                                             \
  })

  DEFINE_BINARY_OP(__add__, [](auto x, auto y) { return x + y; });
  DEFINE_BINARY_OP(__sub__, [](auto x, auto y) { return x - y; });
  DEFINE_BINARY_OP(__mul__, [](auto x, auto y) { return x * y; });
  DEFINE_BINARY_OP(__truediv__, [](auto x, auto y) { return x / y; });
  DEFINE_BINARY_OP(__and__, [](auto x, auto y) { return x && y; });
  DEFINE_BINARY_OP(__or__, [](auto x, auto y) { return x || y; });
  DEFINE_BINARY_OP(__lt__, [](auto x, auto y) { return x < y; });
  DEFINE_BINARY_OP(__le__, [](auto x, auto y) { return x <= y; });
  DEFINE_BINARY_OP(__gt__, [](auto x, auto y) { return x > y; });
  DEFINE_BINARY_OP(__ge__, [](auto x, auto y) { return x >= y; });

#undef DEFINE_BINARY_OP
}
}  // namespace

void BindCommon(py::module *m) {
  BindTarget(m);
  BindType(m);
  BindObject(m);
  BindShared(m);
  BindCinnValue(m);
}
}  // namespace cinn::pybind

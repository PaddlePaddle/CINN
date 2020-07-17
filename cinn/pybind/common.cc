#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {

using common::Object;
using common::Target;
using common::Type;

namespace {
void BindTarget(py::module *);
void BindType(py::module *);
void BindObject(py::module *);
void BindShared(py::module *);

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
  type.def("vector_of", &Type::VectorOf).def("element_of", &Type::ElementOf).def("pointer_of", &Type::PointerOf);

  py::enum_<Type::type_t> type_t(type, "type_t");
  type_t.value("Unk", Type::type_t::Unk)
      .value("Int", Type::type_t::Int)
      .value("UInt", Type::type_t::UInt)
      .value("Float", Type::type_t::Float)
      .value("String", Type::type_t::String)
      .value("Void", Type::type_t::Void)
      .value("Customized", Type::type_t::Customized)
      .export_values();

  py::enum_<Type::cpp_type_t> cpp_type_t(type, "cpp_type_t");
  cpp_type_t.value("None", Type::cpp_type_t::None)
      .value("Const", Type::cpp_type_t::Const)
      .value("Handle", Type::cpp_type_t::Handle)
      .value("HandleHandle", Type::cpp_type_t::HandleHandle)
      .export_values();

  m->def("void", &common::Void)
      .def("int", &common::Int)
      .def("uint", &common::UInt)
      .def("float", &common::Float)
      .def("bool", &common::Bool)
      .def("string", &common::String);

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
  });
}

void BindObject(py::module *m) {
  py::class_<Object, ObjectWrapper> object(*m, "Object");
  object.def("type_info", &Object::type_info);
  //.def_readwrite("ref_count", &Object::__ref_count__);
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
}  // namespace

void BindCommon(py::module *m) {
  BindTarget(m);
  BindType(m);
  BindObject(m);
  BindShared(m);
}
}  // namespace cinn::pybind

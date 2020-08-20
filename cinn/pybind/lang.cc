#include <pybind11/functional.h>

#include <memory>
#include <variant>

#include "cinn/backends/codegen_c.h"
#include "cinn/common/target.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {
using common::Type;
using lang::Placeholder;
using py::arg;
using utils::GetStreamCnt;
using utils::StringFormat;

namespace {
void BindBuffer(py::module *);
void BindLower(py::module *);
void BindPlaceholder(py::module *);
void BindCompute(py::module *);
void BindModule(py::module *);
void BindBuiltin(py::module *);

void BindBuffer(py::module *m) {
  py::class_<lang::Buffer> buffer(*m, "Buffer");
  buffer.def(py::init<ir::Type, const std::string &>(), py::arg("type"), py::arg("name") = "")
      .def(py::init<const ir::Buffer &>())
      .def("buffer", &lang::Buffer::buffer);
}

void BindLower(py::module *m) {
  using py::arg;
  m->def("lower",
         &lang::Lower,
         arg("name"),
         arg("tensor_args"),
         arg("scalar_args")  = std::vector<ir::Var>(),
         arg("temp_tensors") = std::vector<ir::Tensor>(),
         arg("b")            = nullptr);
}

void BindCompute(py::module *m) {
#define MAKE_COMPUTE_FN(__fn)                      \
  py::overload_cast<const std::vector<ir::Expr> &, \
                    __fn,                          \
                    const std::string &,           \
                    const std::vector<ir::Var> &,  \
                    const std::vector<ir::Expr> &>(&lang::Compute)

#define DEFINE_COMPUTE(__fn)                          \
  m->def("compute",                                   \
         MAKE_COMPUTE_FN(__fn),                       \
         arg("domin"),                                \
         arg("fn"),                                   \
         arg("name")        = "",                     \
         arg("reduce_axis") = std::vector<ir::Var>(), \
         arg("shape")       = std::vector<ir::Expr>())

  // DEFINE_COMPUTE(std::function<ir::Expr()>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr)>);
  DEFINE_COMPUTE(std::function<ir::Expr(const std::vector<ir::Expr> &)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr, ir::Expr)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr, ir::Expr, ir::Expr)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr, ir::Expr, ir::Expr, ir::Expr)>);
  DEFINE_COMPUTE(lang::compute_handler_t);

#undef DEFINE_COMPUTE
#undef MAKE_COMPUTE_FN

  py::class_<lang::ReturnType> return_type(*m, "ReturnType");
  return_type.def_readwrite("type", &lang::ReturnType::type)
      .def_readwrite("dims", &lang::ReturnType::dims)
      .def_readwrite("name", &lang::ReturnType::name);

  m->def("call_lowered",
         py::overload_cast<const std::string &, const std::vector<ir::Expr> &, const std::vector<lang::ReturnType> &>(
             &lang::CallLowered));
  m->def("call_extern", py::overload_cast<const std::string &, const std::vector<ir::Expr> &>(&lang::CallExtern));
}

void BindModule(py::module *m) {
  py::class_<lang::Module /*, ir::IrNodeRef*/> module(*m, "Module");

  module.def("target", &lang::Module::target)
      .def("buffers", &lang::Module::buffers)
      .def("functions", &lang::Module::functions)
      .def("submodules", &lang::Module::submodules)
      .def("compile", &lang::Module::Compile)
      .def("get_c_code", [](const lang::Module &self) -> std::string {
        backends::CodeGenC codegen(common::DefaultHostTarget());
        codegen.SetInlineBuiltinCodes(false);
        return codegen.Compile(self, backends::CodeGenC::OutputKind::CImpl);
      });

  py::class_<lang::Module::Builder> builder(module, "Builder");
  builder.def(py::init<const std::string &, const common::Target &>())
      .def("add_function", &lang::Module::Builder::AddFunction)
      .def("add_buffer", &lang::Module::Builder::AddBuffer)
      .def("build", &lang::Module::Builder::Build);
}

class PlaceholderWrapper {
 public:
#define DEFINE_PLACEHOLDER(__dtype, __type) \
  if (dtype == #__dtype) placeholder_ = std::make_unique<Placeholder<__type>>(name, shape)

#define INIT_PLACEHOLDER              \
  DEFINE_PLACEHOLDER(int32, int32_t); \
  DEFINE_PLACEHOLDER(int64, int64_t); \
  DEFINE_PLACEHOLDER(float32, float); \
  DEFINE_PLACEHOLDER(float64, double)

  PlaceholderWrapper(std::string_view dtype, const std::string &name, const std::vector<int> &shape) {
    INIT_PLACEHOLDER;
  }

  PlaceholderWrapper(std::string_view dtype, const std::string &name, const std::vector<ir::Expr> &shape) {
    INIT_PLACEHOLDER;
  }
#undef INIT_PLACEHOLDER
#undef DEFINE_PLACEHOLDER

  ir::Type type() const {
    return std::visit([](auto &v) { return v->type(); }, placeholder_);
  }

  ir::Tensor tensor() const {
    return std::visit([](auto &v) { return v->tensor(); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a) const {
    return std::visit([&](auto &v) { return (*v)(a); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a, ir::Expr b) const {
    return std::visit([&](auto &v) { return (*v)(a, b); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a, ir::Expr b, ir::Expr c) const {
    return std::visit([&](auto &v) { return (*v)(a, b, c); }, placeholder_);
  }

  ir::Expr operator()(const std::vector<ir::Expr> &indices) const {
    return std::visit([&](auto &v) { return (*v)(indices); }, placeholder_);
  }

  operator ir::Tensor() {
    return std::visit([&](auto &v) { return ir::Tensor(*v); }, placeholder_);
  }
  operator ir::Expr() {
    return std::visit([&](auto &v) { return ir::Expr(*v); }, placeholder_);
  }

 private:
  template <typename... Ts>
  using PlaceholderVariant = std::variant<std::unique_ptr<Placeholder<Ts>>...>;

  PlaceholderVariant<int, int64_t, float, double> placeholder_;
};

void BindPlaceholder(py::module *m) {
  py::class_<PlaceholderWrapper> placeholder(*m, "Placeholder");
  placeholder.def(py::init<std::string_view, const std::string &, const std::vector<int> &>())
      .def(py::init<std::string_view, const std::string &, const std::vector<ir::Expr> &>())
      .def("type", &PlaceholderWrapper::type)
      .def("tensor", &PlaceholderWrapper::tensor)
      .def("__call__", [](PlaceholderWrapper &self, ir::Expr a) { return self(std::move(a)); })
      .def("__call__",
           [](PlaceholderWrapper &self, ir::Expr a, ir::Expr b) { return self(std::move(a), std::move(b)); })
      .def("__call__",
           [](PlaceholderWrapper &self, ir::Expr a, ir::Expr b, ir::Expr c) {
             return self(std::move(a), std::move(b), std::move(c));
           })
      .def("__call__", [](PlaceholderWrapper &self, const std::vector<ir::Expr> &indices) { return self(indices); })
      .def("to_expr", [](PlaceholderWrapper &self) { return ir::Expr(self); })
      .def("to_tensor", [](PlaceholderWrapper &self) { return ir::Tensor(self); });

  m->def("create_placeholder", &lang::CreatePlaceHolder);
}

void BindBuiltin(py::module *m) { m->def("sum", &lang::Sum); }
}  // namespace

void BindLang(py::module *m) {
  BindBuffer(m);
  BindLower(m);
  BindPlaceholder(m);
  BindCompute(m);
  BindModule(m);
  BindBuiltin(m);
}
}  // namespace cinn::pybind

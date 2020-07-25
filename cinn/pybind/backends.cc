
#include <pybind11/functional.h>

#include <functional>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/pybind/bind.h"

namespace py = pybind11;

struct cinn_pod_value_t;

namespace cinn::pybind {
using backends::ExecutionEngine;
using backends::ExecutionOptions;
namespace {
void BindExecutionEngine(py::module *);

void BindExecutionEngine(py::module *m) {
  py::class_<ExecutionOptions> options(*m, "ExecutionOptions");
  options.def(py::init<>())
      .def_readwrite("opt_level", &ExecutionOptions::opt_level)
      .def_readwrite("enable_debug_info", &ExecutionOptions::enable_debug_info);

  auto lookup = [](ExecutionEngine &self, std::string_view name) {
    auto *function_ptr    = reinterpret_cast<void (*)(void **, int32_t)>(self.Lookup(name));
    auto function_wrapper = [function_ptr](std::vector<cinn_pod_value_t> &args) {
      function_ptr(reinterpret_cast<void **>(&args.front()), args.size());
    };
    return std::function(function_wrapper);
  };

  py::class_<ExecutionEngine> engine(*m, "ExecutionEngine");
  engine.def_static("create", &ExecutionEngine::Create, py::arg("options") = ExecutionOptions())
      .def(py::init(&ExecutionEngine::Create), py::arg("options") = ExecutionOptions())
      .def("lookup", lookup)
      .def("link", &ExecutionEngine::Link);
}
}  // namespace

void BindBackends(py::module *m) { BindExecutionEngine(m); }
}  // namespace cinn::pybind

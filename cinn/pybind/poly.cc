#include "cinn/poly/stage.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {

using poly::Condition;
using poly::Iterator;
using poly::SplitRestStrategy;
using poly::Stage;
using poly::StageForloopInfo;
using py::arg;

namespace {
void BindMap(py::module *);
void BindStage(py::module *);

void BindMap(py::module *m) {
  py::class_<Iterator> iterator(*m, "Iterator");
  iterator.def_readwrite("id", &Iterator::id)
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const Iterator &>())
      .def("__eq__", [](Iterator &self, Iterator &other) { return self == other; })
      .def("__ne__", [](Iterator &self, Iterator &other) { return self != other; });

  py::class_<Condition> condition(*m, "Condition");
  condition.def_readwrite("cond", &Condition::cond).def(py::init<std::string>()).def("__str__", &Condition::__str__);
}

void BindStage(py::module *m) {
  // enum class SplitRestStrategy
  py::enum_<SplitRestStrategy> split_rest_strategy(*m, "SplitRestStrategy");
  split_rest_strategy.value("kAuto", SplitRestStrategy::kAuto).value("kSeparate", SplitRestStrategy::kSeparate);

  // struct StageForloopInfo
  py::class_<StageForloopInfo> stage_forloop_info(*m, "StageForloopInfo");
  stage_forloop_info.def(py::init<ir::ForType, ir::DeviceAPI>())
      .def_readwrite("for_type", &StageForloopInfo::for_type)
      .def_readwrite("device", &StageForloopInfo::device);

  py::class_<Stage, common::Object> stage(*m, "Stage");
  // enum Stage::ComputeAtKind
  py::enum_<Stage::ComputeAtKind> compute_at_kind(stage, "ComputeAtKind");
  compute_at_kind.value("kComputeAtUnk", Stage::ComputeAtKind::kComputeAtUnk)
      .value("kComputeAtBefore", Stage::ComputeAtKind::kComputeAtBefore)
      .value("kComputeAtAfter", Stage::ComputeAtKind::kComputeAtAfter);

  DefineShared<Stage>(m, "Stage");
  // class Stage
  // stage.def_static("new", &Stage::New, arg("domin"), arg("expr") = ir::Expr(), arg("tensor") = nullptr)
  stage.def("id", &Stage::id)
      .def("expr", &Stage::expr)
      .def("axis", py::overload_cast<int>(&Stage::axis, py::const_))
      .def("axis", py::overload_cast<const std::string &>(&Stage::axis, py::const_))
      .def("axis_names", &Stage::axis_names)
      .def("compute_inline", &Stage::ComputeInline)
      .def("share_buffer_with", &Stage::ShareBufferWith)
      .def("split",
           py::overload_cast<const Iterator &, int, SplitRestStrategy>(&Stage::Split),
           arg("level"),
           arg("factor"),
           arg("strategy") = SplitRestStrategy::kAuto)
      .def("split",
           py::overload_cast<const std::string &, int, SplitRestStrategy>(&Stage::Split),
           arg("level"),
           arg("factor"),
           arg("strategy") = SplitRestStrategy::kAuto)
      .def("split",
           py::overload_cast<int, int, SplitRestStrategy>(&Stage::Split),
           arg("level"),
           arg("factor"),
           arg("strategy") = SplitRestStrategy::kAuto)
      .def("reorder", &Stage::Reorder)
      .def("tile", py::overload_cast<const Iterator &, const Iterator &, int, int>(&Stage::Tile))
      .def("tile", py::overload_cast<int, int, int, int>(&Stage::Tile))
      .def("vectorize", py::overload_cast<int, int>(&Stage::Vectorize))
      .def("vectorize", py::overload_cast<const std::string &, int>(&Stage::Vectorize))
      .def("vectorize", py::overload_cast<const Iterator &, int>(&Stage::Vectorize))
      .def("unroll", py::overload_cast<int>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const std::string &>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const Iterator &>(&Stage::Unroll))
      .def("compute_at", &Stage::ComputeAtSchedule, arg("other"), arg("level"), arg("kind") = Stage::kComputeAtUnk)
      .def("skew", &Stage::Skew)
      // TODO(fuchang01): GpuThreads
      // TODO(fuchang01): GpuBlocks
      .def("ctrl_depend", &Stage::CtrlDepend);
}
}  // namespace

void BindPoly(py::module *m) {
  BindMap(m);
  BindStage(m);
}

}  // namespace cinn::pybind

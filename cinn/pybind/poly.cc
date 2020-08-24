#include "cinn/poly/stage.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {

using poly::Condition;
using poly::Iterator;
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
  py::class_<Stage, common::Object> stage(*m, "Stage");
  // enum Stage::ComputeAtKind
  py::enum_<Stage::ComputeAtKind> compute_at_kind(stage, "ComputeAtKind");
  compute_at_kind.value("kComputeAtUnk", Stage::ComputeAtKind::kComputeAtAuto)
      .value("kComputeAtBefore", Stage::ComputeAtKind::kComputeAtBefore)
      .value("kComputeAtAfter", Stage::ComputeAtKind::kComputeAtAfter);

  DefineShared<Stage>(m, "Stage");
  stage.def("id", &Stage::id)
      .def("expr", &Stage::expr)
      .def("axis", py::overload_cast<int>(&Stage::axis, py::const_))
      .def("axis", py::overload_cast<const std::string &>(&Stage::axis, py::const_))
      .def("axis_names", &Stage::axis_names)
      .def("bind", &Stage::Bind)
      .def("compute_inline", &Stage::ComputeInline)
      .def("share_buffer_with", &Stage::ShareBufferWith)
      .def("split", py::overload_cast<const Iterator &, int>(&Stage::Split), arg("level"), arg("factor"))
      .def("split", py::overload_cast<const std::string &, int>(&Stage::Split), arg("level"), arg("factor"))
      .def("split", py::overload_cast<int, int>(&Stage::Split), arg("level"), arg("factor"))
      .def("fuse", py::overload_cast<int, int>(&Stage::Fuse), arg("level0"), arg("level1"))
      .def("reorder", &Stage::Reorder)
      .def("tile", py::overload_cast<const Iterator &, const Iterator &, int, int>(&Stage::Tile))
      .def("tile", py::overload_cast<int, int, int, int>(&Stage::Tile))
      .def("vectorize", py::overload_cast<int, int>(&Stage::Vectorize))
      .def("vectorize", py::overload_cast<const std::string &, int>(&Stage::Vectorize))
      .def("vectorize", py::overload_cast<const Iterator &, int>(&Stage::Vectorize))
      .def("unroll", py::overload_cast<int>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const std::string &>(&Stage::Unroll))
      .def("unroll", py::overload_cast<const Iterator &>(&Stage::Unroll))
      .def("compute_at", &Stage::ComputeAtSchedule, arg("other"), arg("level"), arg("kind") = Stage::kComputeAtAuto)
      .def("skew", &Stage::Skew)
      .def("ctrl_depend", &Stage::CtrlDepend)
      .def("cache_read", &Stage::CacheRead)
      .def("cache_write", &Stage::CacheRead);
}

void BindStageMap(py::module *m) {
  DefineShared<poly::_StageMap_>(m, "StageMap");
  py::class_<poly::StageMap, Shared<poly::_StageMap_>> stage_map(*m, "StageMap");
  m->def("create_stages", &poly::CreateStages, py::arg("tensors"));
}

}  // namespace

void BindPoly(py::module *m) {
  BindMap(m);
  BindStage(m);
  BindStageMap(m);
}

}  // namespace cinn::pybind

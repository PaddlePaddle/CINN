#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cinn::pybind {

void BindRuntime(pybind11::module *m);
void BindCommon(pybind11::module *m);
void BindLang(pybind11::module *m);
void BindIr(pybind11::module *m);
void BindBackends(pybind11::module *m);
void BindPoly(pybind11::module *m);
void BindOptim(pybind11::module *m);

}  // namespace cinn::pybind

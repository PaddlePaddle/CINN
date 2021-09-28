#pragma once

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <absl/types/variant.h>

namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<absl::variant<Ts...>> : variant_caster<absl::variant<Ts...>> {};

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct type_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};

template <>
struct type_caster<absl::string_view> : string_caster<absl::string_view, true> {};
}  // namespace detail
}  // namespace pybind11

namespace cinn::pybind {

void BindRuntime(pybind11::module *m);
void BindCommon(pybind11::module *m);
void BindLang(pybind11::module *m);
void BindIr(pybind11::module *m);
void BindBackends(pybind11::module *m);
void BindPoly(pybind11::module *m);
void BindOptim(pybind11::module *m);
void BindPE(pybind11::module *m);
void BindFrontend(pybind11::module *m);
void BindFramework(pybind11::module *m);

}  // namespace cinn::pybind

#include "cinn/frontend/paddle/cpp/op_desc.h"

#include <cstdint>
#include <utility>

namespace cinn::frontend::paddle::cpp {

#define SET_ATTR_IMPL(T, repr__)                                 \
  template <>                                                    \
  void OpDesc::SetAttr<T>(const std::string& name, const T& v) { \
    attr_types_[name] = AttrType::repr__;                        \
    attrs_[name]      = v;                                       \
  }

SET_ATTR_IMPL(int32_t, INT);
SET_ATTR_IMPL(float, FLOAT);
SET_ATTR_IMPL(std::string, STRING);
SET_ATTR_IMPL(bool, BOOLEAN);
SET_ATTR_IMPL(int64_t, LONG);
SET_ATTR_IMPL(std::vector<int>, INTS);
SET_ATTR_IMPL(std::vector<float>, FLOATS);
SET_ATTR_IMPL(std::vector<std::string>, STRINGS);
SET_ATTR_IMPL(std::vector<int64_t>, LONGS);

std::pair<OpDesc::attrs_t::const_iterator, OpDesc::attr_types_t::const_iterator> FindAttr(const OpDesc& desc,
                                                                                          const std::string& name) {
  auto it = desc.attrs().find(name);
  CHECK(it != desc.attrs().end()) << "No attributes called " << name << " found";
  auto attr_it = desc.attr_types().find(name);
  CHECK(attr_it != desc.attr_types().end());
  return std::make_pair(it, attr_it);
}

#define GET_IMPL_ONE(T, repr__)                                                                \
  template <>                                                                                  \
  T OpDesc::GetAttr<T>(const std::string& name) const {                                        \
    auto pair = FindAttr(*this, name);                                                         \
    CHECK(pair.second->second == AttrType::repr__)                                             \
        << "The attrbute [" << pair.second->first << "]'s type doesn't match the target type!" \
        << "Its type should be " << #repr__ << ". Please check.";                              \
    return std::any_cast<T>(pair.first->second);                                               \
  }

GET_IMPL_ONE(int32_t, INT)
std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : outputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::input_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : InputArgumentNames()) {
    for (auto& vars : Input(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::output_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : OutputArgumentNames()) {
    for (auto& vars : Output(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::InputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : inputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::Input(const std::string& param) const {
  auto it = inputs_.find(param);
  CHECK(it != inputs_.end());
  return it->second;
}

std::vector<std::string> OpDesc::Output(const std::string& param) const {
  auto it = outputs_.find(param);
  CHECK(it != outputs_.end());
  return it->second;
}

bool OpDesc::HasOutput(const std::string& param) const {
  auto it = outputs_.find(param);
  return it != outputs_.end();
}

GET_IMPL_ONE(float, FLOAT);
GET_IMPL_ONE(std::string, STRING);
GET_IMPL_ONE(int64_t, LONG);
GET_IMPL_ONE(bool, BOOLEAN);
GET_IMPL_ONE(std::vector<int64_t>, LONGS);
GET_IMPL_ONE(std::vector<float>, FLOATS);
GET_IMPL_ONE(std::vector<int>, INTS);
GET_IMPL_ONE(std::vector<std::string>, STRINGS);

}  // namespace cinn::frontend::paddle::cpp

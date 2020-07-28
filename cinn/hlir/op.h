#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cinn/utils/any.h"
#include "cinn/utils/base.h"
#include "cinn/utils/registry.h"

using dmlc::any;

namespace cinn {
namespace hlir {
class Op;

struct OpRegistry {
  std::recursive_mutex mutex;
  std::atomic<int> op_counter{0};
  std::unordered_map<std::string, std::unique_ptr<any>> attr;
  static OpRegistry* Global() {
    static OpRegistry inst;
    return &inst;
  }
};

template <typename ValueType>
class OpValueType {
 public:
  inline const ValueType& operator[](const Op* op) const {
    CHECK(op != nullptr) << "The input op is nullptr and it is invalid! Please check again.";
    const uint32_t idx = op->index;
    CHECK(idx < data.size()) << "Attribute " << attr_name << " has not been registered for Operator " << op->name;
    return data[idx];
  }

  inline const ValueType& get(const Op* op, const ValueType& def_value) const {
    if (op == nullptr) return def_value;
    const uint32_t idx = op->index;
    if (idx < data.size()) {
      return data[idx];
    } else {
      return def_value;
    }
  }

  inline bool find(const Op* op) const {
    if (op == nullptr) return false;
    const uint32_t idx = op->index;
    return idx < data.size();
  }

 private:
  friend class Op;
  std::string attr_name;
  std::vector<ValueType> data;
  OpValueType() = default;
};

class Op {
 public:
  std::string name;
  std::string description;
  uint32_t num_inputs    = 1;
  uint32_t num_outputs   = 1;
  uint32_t support_level = 10;

  inline Op& describe(const std::string description) {
    this->description = description;
    return *this;
  }

  inline Op& set_num_inputs(uint32_t n) {
    this->num_inputs = n;
    return *this;
  }

  inline Op& set_num_outputs(uint32_t n) {
    this->num_outputs = n;
    return *this;
  }

  inline Op& set_support_level(uint32_t n) {
    this->support_level = n;
    return *this;
  }
  /*!
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * \param op_name Name of the operator.
   * \return Pointer to a Op, valid throughout program lifetime.
   */
  static const Op* Get(const std::string& op_name) {
    const Op* op = dmlc::Registry<Op>::Find(op_name);
    CHECK(op != nullptr) << "Operator " << op_name << " is not registered";
    return op;
  }

  template <typename ValueType>
  inline Op& set_attr(const std::string& attr_name, const ValueType& value) {
    UpdateAttrMap(attr_name, [this, attr_name, value](any* pmap) {
      if (pmap->empty()) {
        OpValueType<ValueType> pm;
        pm.attr_name = attr_name;
        *pmap        = std::move(pm);
      }
      std::vector<ValueType>& vec = dmlc::get<OpValueType<ValueType>>(*pmap).data;
      // resize the value type.
      if (vec.size() <= index) {
        vec.resize(index + 1, ValueType());
      }
      vec[index] = value;
    });
    return *this;
  }
  template <typename ValueType>
  static const OpValueType<ValueType>& GetAttr(const std::string& attr_name) {
    const any* ref = GetAttrMap(attr_name);
    if (ref == nullptr) {
      // update the attribute map of the key by creating new empty OpMap
      UpdateAttrMap(attr_name, [attr_name](any* pmap) {
        if (pmap->empty()) {
          OpValueType<ValueType> pm;
          pm.attr_name = attr_name;
          *pmap        = std::move(pm);
        }
      });
      ref = GetAttrMap(attr_name);
    }
    return dmlc::get<OpValueType<ValueType>>(*ref);
  }

 private:
  template <typename ValueType>
  friend class OpValueType;
  friend class dmlc::Registry<Op>;
  uint32_t index{0};
  Op() {
    OpRegistry* reg = OpRegistry::Global();
    index           = reg->op_counter++;
  }
  static const any* GetAttrMap(const std::string& key) {
    auto& dict = OpRegistry::Global()->attr;
    auto it    = dict.find(key);
    if (it != dict.end()) {
      return it->second.get();
    } else {
      return nullptr;
    }
  }
  // update the attribute OpValueType
  static void UpdateAttrMap(const std::string& key, std::function<void(any*)> updater) {
    OpRegistry* reg = OpRegistry::Global();
    std::lock_guard<std::recursive_mutex>(reg->mutex);
    std::unique_ptr<any>& value = reg->attr[key];
    if (value.get() == nullptr) value.reset(new any());
    if (updater != nullptr) updater(value.get());
  }
};

// internal macros to make
#define HLIR_REGISTER_VAR_DEF(OpName) static DMLC_ATTRIBUTE_UNUSED ::cinn::hlir::Op& __make_##HlirOp##_##OpName

/*!
 * \def HLIR_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * \param OpName The name of registry
 *
 * \code
 *
 *  HLIR_REGISTER_OP(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define HLIR_REGISTER_OP(OpName)                                \
  DMLC_STR_CONCAT(HLIR_REGISTER_VAR_DEF(OpName), __COUNTER__) = \
      ::dmlc::Registry<::cinn::hlir::Op>::Get()->__REGISTER_OR_GET__(#OpName)
}  // namespace hlir
}  // namespace cinn

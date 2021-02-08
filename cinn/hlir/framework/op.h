#pragma once
#include <glog/logging.h>

#include <any>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/utils/registry.h"

namespace cinn {
namespace hlir {
namespace framework {
class Operator;

using shape_t = std::vector<int32_t>;
using dim_t   = shape_t ::value_type;

/*! \brief operator pattern used in graph fusion */
enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // The pattern for tuple nodes. Can fuse into subsequent injective ops,
  // but treated specially
  kTuple = 7,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

struct OpRegistry : public Registry<Operator> {
  std::recursive_mutex mutex;
  std::atomic<int> op_counter{0};
  std::unordered_map<std::string, std::unique_ptr<std::any>> attrs;

  static OpRegistry* Global() {
    static OpRegistry x;
    return &x;
  }

 private:
  OpRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(OpRegistry);
};

template <typename ValueType>
class OpValueType {
 public:
  inline const ValueType& operator[](const Operator* op) const;

  inline const ValueType& Get(const Operator* op, const ValueType& def_value) const;

  inline bool Find(const Operator* op) const;

 private:
  friend class Operator;
  std::string attr_name;
  std::vector<ValueType> data;
  OpValueType() = default;
};

class Operator {
 public:
  std::string name;
  std::string description;
  uint32_t num_inputs{1};
  uint32_t num_outputs{1};
  uint32_t support_level{10};

  inline Operator& describe(const std::string description) {
    this->description = description;
    return *this;
  }

  inline Operator& set_num_inputs(uint32_t n) {
    this->num_inputs = n;
    return *this;
  }

  inline Operator& set_num_outputs(uint32_t n) {
    this->num_outputs = n;
    return *this;
  }

  inline Operator& set_support_level(uint32_t n) {
    this->support_level = n;
    return *this;
  }
  /**
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * @param op_name Name of the operator.
   * @return Pointer to a Op, valid throughout program lifetime.
   */
  static const Operator* Get(const std::string& op_name) {
    const Operator* op = OpRegistry::Global()->Find(op_name);
    CHECK(op) << "Operator [" << op_name << "] is not registered";
    return op;
  }

  template <typename ValueType>
  inline Operator& set_attr(const std::string& attr_name, const ValueType& value) {
    UpdateAttrMap(attr_name, [this, attr_name, value](std::any* pmap) {
      if (!pmap->has_value()) {
        OpValueType<ValueType> pm;
        pm.attr_name = attr_name;
        *pmap        = std::move(pm);
      }
      std::vector<ValueType>& vec = std::any_cast<OpValueType<ValueType>&>(*pmap).data;
      // resize the value type.
      if (vec.size() <= index) {
        vec.resize(index + 1, ValueType());
      }
      vec[index] = value;
    });
    return *this;
  }
  template <typename ValueType>
  static const OpValueType<ValueType>& GetAttrs(const std::string& attr_name) {
    const std::any* ref = GetAttrMap(attr_name);
    if (ref == nullptr) {
      //! update the attribute map of the key by creating new empty OpMap
      UpdateAttrMap(attr_name, [attr_name](std::any* pmap) {
        if (!pmap->has_value()) {
          OpValueType<ValueType> pm;
          pm.attr_name = attr_name;
          *pmap        = std::move(pm);
        }
      });
      ref = GetAttrMap(attr_name);
    }
    return std::any_cast<const OpValueType<ValueType>&>(*ref);
  }

  auto get_index() const { return index; }

 private:
  template <typename ValueType>
  friend class OpValueType;
  friend class Registry<Operator>;
  uint32_t index{0};
  Operator() { index = OpRegistry::Global()->op_counter++; }
  static const std::any* GetAttrMap(const std::string& key) {
    auto& dict = OpRegistry::Global()->attrs;
    auto it    = dict.find(key);
    if (it != dict.end()) {
      return it->second.get();
    } else {
      return nullptr;
    }
  }
  //! update the attribute OpValueType
  static void UpdateAttrMap(const std::string& key, std::function<void(std::any*)> updater) {
    OpRegistry* reg = OpRegistry::Global();
    std::lock_guard<std::recursive_mutex>(reg->mutex);
    std::unique_ptr<std::any>& value = reg->attrs[key];
    if (value.get() == nullptr) value.reset(new std::any());
    if (updater != nullptr) updater(value.get());
  }
};

template <typename ValueType>
const ValueType& OpValueType<ValueType>::operator[](const Operator* op) const {
  CHECK(op) << "The input op is nullptr and it is invalid! Please check again.";
  const uint32_t idx = op->index;
  CHECK_LT(idx, data.size()) << "Attribute " << attr_name << " has not been registered for Operator " << op->name;
  return data[idx];
}

template <typename ValueType>
const ValueType& OpValueType<ValueType>::Get(const Operator* op, const ValueType& def_value) const {
  if (!op) return def_value;
  const uint32_t idx = op->index;
  if (idx < data.size()) {
    return data[idx];
  } else {
    return def_value;
  }
}

template <typename ValueType>
bool OpValueType<ValueType>::Find(const Operator* op) const {
  if (!op) return false;
  const uint32_t idx = op->index;
  return idx < data.size();
}

// internal macros to make
#define CINN_REGISTER_VAR_DEF(OpName) static ::cinn::hlir::framework::Operator& __make_##HlirOp##_##OpName

/**
 * @def CINNR_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * @param OpName The name of registry
 *
 * \code
 *  CINN_REGISTER_OP(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 * \endcode
 */
#define CINN_REGISTER_OP(OpName)                                \
  CINN_STR_CONCAT(CINN_REGISTER_VAR_DEF(OpName), __COUNTER__) = \
      ::cinn::hlir::framework::OpRegistry::Global()->__REGISTER_OR_GET__(#OpName)

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

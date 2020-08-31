#pragma once
#include <glog/logging.h>
#include <algorithm>
#include <map>
#include <variant>

#include <string>
#include <vector>
#include "cinn/frontend/paddle/framework.pb.h"

namespace cinn {
namespace frontend {
namespace paddle {
using namespace ::paddle::framework;  // NOLINT

enum class PrecisionType : int {
  kUnk   = 0,
  kFloat = 1,
  kInt8  = 2,
  kInt32 = 3,
  kAny   = 4,  // any precision
  kFP16  = 5,
  kBool  = 6,
  kInt64 = 7,
  kInt16 = 8,
  NUM    = 9,  // number of fields.
};

// The AttrType is used to make the proto::AttrType portable.
enum class OpAttrType {
  INT      = 0,
  FLOAT    = 1,
  STRING   = 2,
  INTS     = 3,
  FLOATS   = 4,
  STRINGS  = 5,
  BOOLEAN  = 6,
  BOOLEANS = 7,
  BLOCK    = 8,
  LONG     = 9,
  BLOCKS   = 10,
  LONGS    = 11,
  UNK,
};

enum class VarDataType {
  // Pod Types
  BOOL = 0,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  // Tensor<size_t> is used in C++.
  SIZE_T,
  UINT8,
  INT8,

  // Other types that may need additional descriptions
  LOD_TENSOR,
  SELECTED_ROWS,
  FEED_MINIBATCH,
  FETCH_LIST,
  STEP_SCOPES,
  LOD_RANK_TABLE,
  LOD_TENSOR_ARRAY,
  PLACE_LIST,
  READER,
  // Any runtime decided variable type is raw
  // raw variables should manage their own allocations
  // in operators like nccl_op
  RAW,
  TUPLE
};

inline VarDataType ConvertPrecisionType(PrecisionType type) {
#define CASE(ptype, vtype)      \
  case PrecisionType::k##ptype: \
    return VarDataType::vtype;  \
    break
  switch (type) {
    CASE(Float, FP32);
    CASE(Int8, INT8);
    CASE(Int32, INT32);
    CASE(FP16, FP16);
    CASE(Bool, BOOL);
    CASE(Int64, INT64);
    CASE(Int16, INT16);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return VarDataType();
  }
#undef CASE
}

inline PrecisionType ConvertPrecisionType(VarDataType type) {
#define CASE(ptype, vtype)          \
  case VarDataType::vtype:          \
    return PrecisionType::k##ptype; \
    break
  switch (type) {
    CASE(Float, FP32);
    CASE(Int8, INT8);
    CASE(Int32, INT32);
    CASE(FP16, FP16);
    CASE(Bool, BOOL);
    CASE(Int64, INT64);
    CASE(Int16, INT16);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return PrecisionType();
  }
#undef CASE
}

class OpDescReadAPI {
 public:
  virtual std::string Type() const                                        = 0;
  virtual std::vector<std::string> Input(const std::string &param) const  = 0;
  virtual std::vector<std::string> InputArgumentNames() const             = 0;
  virtual std::vector<std::string> Output(const std::string &param) const = 0;
  virtual std::vector<std::string> OutputArgumentNames() const            = 0;
  virtual bool HasAttr(const std::string &name) const                     = 0;
  virtual OpAttrType GetAttrType(const std::string &name) const           = 0;
  virtual std::vector<std::string> AttrNames() const                      = 0;

  template <typename T>
  T GetAttr(const std::string &name) const;

  std::string Repr() const;

  virtual ~OpDescReadAPI() = default;
};

class OpDescWriteAPI {
 public:
  virtual void SetType(const std::string &type) { NotImplemented(); }
  virtual void SetInput(const std::string &param, const std::vector<std::string> &args) { NotImplemented(); }
  virtual void SetOutput(const std::string &param, const std::vector<std::string> &args) { NotImplemented(); }

  template <typename T>
  void SetAttr(const std::string &name, const T &v) {
    NotImplemented();
  }

  virtual ~OpDescWriteAPI() = default;

 private:
  void NotImplemented() const { LOG(FATAL) << "OpDescWriteAPI is not available in model read-only mode."; }
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class OpDescAPI : public OpDescReadAPI, public OpDescWriteAPI {
 public:
  using AttrType       = OpAttrType;
  virtual ~OpDescAPI() = default;
};

using Attribute       = std::variant<int, float, bool, std::vector<std::string>, std::vector<int>>;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;

/*
 * The lite::OpDesc, an light-weight implementation of wrapper of proto::OpDesc.
 * Unlike the original one in framework::OpDesc, we remove the local members
 * except the desc_, to avoid the inconsistent state, which is normal in the
 * original interface and results in bugs.
 */
class OpDesc : public OpDescAPI {
 public:
  OpDesc() = delete;

  explicit OpDesc(::paddle::framework::proto::OpDesc *desc) : desc_(desc) { CHECK(desc_); }

  proto::OpDesc *Proto() { return desc_; }
  const proto::OpDesc &ReadonlyProto() const { return *desc_; }

  std::string Type() const override { return desc_->type(); }

  void SetType(const std::string &type) override { desc_->set_type(type); }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string &param) const override {
    return GetArguments(desc_->inputs(), param);
  }

  std::vector<std::string> InputArgumentNames() const override { return GetArgumentNames(desc_->inputs()); }

  void SetInput(const std::string &param, const std::vector<std::string> &args) override {
    SetArgument(desc_->mutable_inputs(), param, args);
  }

  std::vector<std::string> Output(const std::string &param) const override {
    return GetArguments(desc_->outputs(), param);
  }

  std::vector<std::string> OutputArgumentNames() const override { return GetArgumentNames(desc_->outputs()); }

  void SetOutput(const std::string &param, const std::vector<std::string> &args) override {
    SetArgument(desc_->mutable_outputs(), param, args);
  }

  bool HasAttr(const std::string &name) const override {
    const auto &xs = desc_->attrs();
    auto it        = std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc_Attr &x) { return x.name() == name; });
    return it != xs.end();
  }

  AttrType GetAttrType(const std::string &name) const override {
    const auto &xs = desc_->attrs();
    auto it        = std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc_Attr &x) { return x.name() == name; });
    CHECK(it != xs.end());
#define DEF_ONE(type__)         \
  case proto::AttrType::type__: \
    return AttrType::type__;

    switch (it->type()) {
      DEF_ONE(INT);
      DEF_ONE(FLOAT);
      DEF_ONE(STRING);
      DEF_ONE(INTS);
      DEF_ONE(FLOATS);
      DEF_ONE(STRINGS);
      DEF_ONE(BOOLEAN);
      DEF_ONE(BOOLEANS);
      DEF_ONE(BLOCK);
      DEF_ONE(LONG);
      DEF_ONE(BLOCKS);
      DEF_ONE(LONGS);
      default:
        LOG(FATAL) << "Unknown attribute type";
        return static_cast<AttrType>(-1);
    }
#undef DEF_ONE
  }

  std::vector<std::string> AttrNames() const override {
    std::vector<std::string> res;
    const auto &xs = desc_->attrs();
    std::transform(xs.begin(), xs.end(), std::back_inserter(res), [](const proto::OpDesc_Attr &x) { return x.name(); });
    return res;
  }

  template <typename T>
  void SetAttr(const std::string &name, const T &v);

  template <typename T>
  T GetAttr(const std::string &name) const;

 private:
  std::vector<std::string> GetArguments(const google::protobuf::RepeatedPtrField<proto::OpDesc_Var> &xs,
                                        const std::string &param) const {
    std::vector<std::string> res;
    auto it = std::find_if(xs.begin(), xs.end(), [&](const proto::OpDesc_Var &it) { return it.parameter() == param; });
    CHECK(it != xs.end());

    const auto &ys = it->arguments();
    std::transform(ys.begin(), ys.end(), std::back_inserter(res), [](const std::string &x) { return x; });
    return res;
  }

  void SetArgument(google::protobuf::RepeatedPtrField<proto::OpDesc_Var> *xs,
                   const std::string &param,
                   const std::vector<std::string> &args) {
    auto it =
        std::find_if(xs->begin(), xs->end(), [&](const proto::OpDesc_Var &it) { return it.parameter() == param; });
    if (it == xs->end()) {
      auto *new_arg = xs->Add();
      new_arg->set_parameter(param);
      for (const auto &arg : args) {
        *new_arg->mutable_arguments()->Add() = arg;
      }
    } else {
      it->mutable_arguments()->Clear();
      for (const auto &arg : args) {
        *it->mutable_arguments()->Add() = arg;
      }
    }
  }

  std::vector<std::string> GetArgumentNames(const google::protobuf::RepeatedPtrField<proto::OpDesc_Var> &xs) const {
    std::vector<std::string> res;
    std::transform(
        xs.begin(), xs.end(), std::back_inserter(res), [](const proto::OpDesc_Var &x) { return x.parameter(); });
    return res;
  }

 private:
  proto::OpDesc *desc_;
};

template <>
void OpDesc::SetAttr<std::string>(const std::string &name, const std::string &v);

template <>
void OpDesc::SetAttr<std::vector<int>>(const std::string &name, const std::vector<int> &v);

}  // namespace paddle
}  // namespace frontend
}  // namespace cinn

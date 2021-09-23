#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/frontend/paddle/compatible_pb.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace cinn::frontend::paddle {
namespace pb    = ::paddle::framework;
using PbVarType = pb::proto::VarType;

namespace utils {

cpp::VarDescAPI::Type TransformVarTypePbToCpp(const PbVarType::Type &type) {
#define SET_TYPE_CASE_ITEM(type__)        \
  case PbVarType::type__:                 \
    return cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
  }
#undef SET_TYPE_CASE_ITEM
}

PbVarType::Type TransformVarTypeCppToPb(const cpp::VarDescAPI::Type &type) {
#define SET_TYPE_CASE_ITEM(type__)    \
  case cpp::VarDescAPI::Type::type__: \
    return PbVarType::type__;         \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
  }
#undef SET_TYPE_CASE_ITEM
}

cpp::VarDescAPI::Type TransformVarDataTypePbToCpp(const PbVarType::Type &type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)   \
  case PbVarType::type__:                 \
    return cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      LOG(FATAL) << "Unknown var data type";
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

PbVarType::Type TransformVarDataTypeCppToPb(const cpp::VarDescAPI::Type &type) {
#define SET_DATA_TYPE_CASE_ITEM(type__) \
  case cpp::VarDescAPI::Type::type__:   \
    return PbVarType::type__;           \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      LOG(FATAL) << "Unknown var data type";
  }
#undef SET_DATA_TYPE_CASE_ITEM
}
}  // namespace utils

template <>
void TransformVarDescAnyToCpp<pb::VarDesc>(pb::VarDesc *pb_desc, cpp::VarDesc *cpp_desc) {
  cpp_desc->SetName(pb_desc->Name());
  cpp_desc->SetType(utils::TransformVarTypePbToCpp(pb_desc->GetType()));
  cpp_desc->SetPersistable(pb_desc->Persistable());
  if (pb_desc->Name() != "feed" && pb_desc->Name() != "fetch") {
    cpp_desc->SetDataType(utils::TransformVarDataTypePbToCpp(pb_desc->GetDataType()));
    cpp_desc->SetShape(pb_desc->GetShape());
  }
}

template <>
void TransformVarDescCppToAny<pb::VarDesc>(const cpp::VarDesc &cpp_desc, pb::VarDesc *pb_desc) {
  pb_desc->SetName(cpp_desc.Name());
  pb_desc->SetType(utils::TransformVarTypeCppToPb(cpp_desc.GetType()));
  pb_desc->SetPersistable(cpp_desc.Persistable());
  if (cpp_desc.Name() != "feed" && cpp_desc.Name() != "fetch") {
    pb_desc->SetShape(cpp_desc.GetShape());
    pb_desc->SetDataType(utils::TransformVarDataTypeCppToPb(cpp_desc.GetDataType()));
  }
}

/// For OpDesc transform
void OpInputsPbToCpp(pb::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc->InputArgumentNames()) {
    cpp_desc->SetInput(param, pb_desc->Input(param));
  }
}

void OpInputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    pb_desc->SetInput(param, cpp_desc.Input(param));
  }
}

void OpOutputsPbToCpp(pb::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : pb_desc->OutputArgumentNames()) {
    cpp_desc->SetOutput(param, pb_desc->Output(param));
  }
}

void OpOutputsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    pb_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

void OpAttrsPbToCpp(pb::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  using AttrType = pb::proto::AttrType;
  auto set_attr  = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                                        \
  case AttrType::type__:                                           \
    cpp_desc->SetAttr<T>(name, pb_desc->GetAttrIfExists<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      IMPL_ONE(LONG, int64_t);
      IMPL_ONE(LONGS, std::vector<int64_t>);
      case AttrType::BLOCK: {
        auto i = pb_desc->GetAttrIfExists<int16_t>(name);
        cpp_desc->SetAttr<int32_t>(name, i);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported attr type found " << static_cast<int>(type);
    }
  };
#undef IMPL_ONE

  for (const auto &attr_name : pb_desc->AttrNames()) {
    auto type = pb_desc->GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

void OpAttrsCppToPb(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  using AttrType = cpp::OpDescAPI::AttrType;
  auto set_attr  = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                            \
  case AttrType::type__:                               \
    pb_desc->SetAttr(name, cpp_desc.GetAttr<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      IMPL_ONE(LONG, int64_t);
      IMPL_ONE(LONGS, std::vector<int64_t>);
      default:
        LOG(FATAL) << "Unsupported attr type found: " << static_cast<int>(type);
    }
  };
#undef IMPL_ONE

  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

template <>
void TransformOpDescAnyToCpp<pb::OpDesc>(pb::OpDesc *pb_desc, cpp::OpDesc *cpp_desc) {
  cpp_desc->SetType(pb_desc->Type());
  OpInputsPbToCpp(pb_desc, cpp_desc);
  OpOutputsPbToCpp(pb_desc, cpp_desc);
  OpAttrsPbToCpp(pb_desc, cpp_desc);
}

template <>
void TransformOpDescCppToAny<pb::OpDesc>(const cpp::OpDesc &cpp_desc, pb::OpDesc *pb_desc) {
  pb_desc->SetType(cpp_desc.Type());
  OpInputsCppToPb(cpp_desc, pb_desc);
  OpOutputsCppToPb(cpp_desc, pb_desc);
  OpAttrsCppToPb(cpp_desc, pb_desc);
}

/// For BlockDesc transform
template <>
void TransformBlockDescAnyToCpp<pb::BlockDesc>(pb::BlockDesc *pb_desc, cpp::BlockDesc *cpp_desc) {
  cpp_desc->SetIdx(pb_desc->ID());
  cpp_desc->SetParentIdx(pb_desc->Parent());
  cpp_desc->SetForwardBlockIdx(pb_desc->ForwardBlockID());

  cpp_desc->ClearOps();
  const auto &all_ops = pb_desc->AllOps();
  for (const auto &op : all_ops) {
    auto *cpp_op_desc = cpp_desc->AddOp<cpp::OpDesc>();
    TransformOpDescAnyToCpp(*op, cpp_op_desc);
  }

  cpp_desc->ClearVars();
  const auto &all_vars = pb_desc->AllVars();
  for (const auto &var : all_vars) {
    auto *cpp_var_desc = cpp_desc->AddVar<cpp::VarDesc>();
    TransformVarDescAnyToCpp(var, cpp_var_desc);
  }
}

template <>
void TransformBlockDescCppToAny<pb::BlockDesc>(const cpp::BlockDesc &cpp_desc, pb::BlockDesc *pb_desc) {
  pb_desc->Proto()->Clear();

  pb_desc->Proto()->set_idx(cpp_desc.Idx());
  pb_desc->Proto()->set_parent_idx(cpp_desc.ParentIdx());
  pb_desc->Proto()->set_forward_block_idx(cpp_desc.ForwardBlockIdx());

  for (size_t i = 0; i < cpp_desc.OpsSize(); ++i) {
    const auto &cpp_op_desc = cpp_desc.template GetConstOp<cpp::OpDesc>(static_cast<int32_t>(i));
    auto *pb_op_desc        = pb_desc->AppendOp();
    TransformOpDescCppToAny(cpp_op_desc, pb_op_desc);
  }

  for (size_t i = 0; i < cpp_desc.VarsSize(); ++i) {
    const auto &cpp_var_desc = cpp_desc.template GetConstVar<cpp::VarDesc>(static_cast<int32_t>(i));
    auto *pb_var_desc        = pb_desc->Var(cpp_var_desc.Name());
    TransformVarDescCppToAny(cpp_var_desc, pb_var_desc);
  }
}

/// For ProgramDesc transform
template <>
void TransformProgramDescAnyToCpp<pb::ProgramDesc>(pb::ProgramDesc *pb_desc, cpp::ProgramDesc *cpp_desc) {
  if (pb_desc->Proto()->version().has_version()) {
    cpp_desc->SetVersion(pb_desc->Version());
  }

  cpp_desc->ClearBlocks();
  for (size_t i = 0; i < pb_desc->Size(); ++i) {
    const auto &pb_block_desc = pb_desc->Block(i);
    auto *cpp_block_desc      = cpp_desc->AddBlock<cpp::BlockDesc>();
    TransformBlockDescAnyToCpp(pb_block_desc, cpp_block_desc);
  }
}

template <>
void TransformProgramDescCppToAny<pb::ProgramDesc>(const cpp::ProgramDesc &cpp_desc, pb::ProgramDesc *pb_desc) {
  pb_desc->Proto()->Clear();

  if (cpp_desc.HasVersion()) {
    pb_desc->SetVersion(cpp_desc.Version());
  }

  // For paddle proto program, the only way to add block is invoke AppendBlock(),
  // the AppendBlock need one necessary parameter: const BlockDesc &parent,
  // but the only function of parent is set the block's parent_idx value.
  // That's why here we create a fake block and only set it's id zero;
  pb::ProgramDesc fake_prog;
  pb::BlockDesc fake_block(&fake_prog, fake_prog.Proto()->add_blocks());
  // In fact, the is is not important, because in TransformBlockDescCppToAny,
  // the parent_idx will reset, so here all is fake.
  fake_block.Proto()->set_idx(0);

  for (size_t i = 0; i < cpp_desc.BlocksSize(); ++i) {
    const auto &cpp_block_desc = cpp_desc.GetConstBlock<cpp::BlockDesc>(i);
    auto pb_block_desc         = pb_desc->AppendBlock(fake_block);
    TransformBlockDescCppToAny(cpp_block_desc, &pb_block_desc);
  }
}

}  // namespace cinn::frontend::paddle

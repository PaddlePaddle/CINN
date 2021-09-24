#pragma once
#include "cinn/frontend/paddle/cpp/desc_api.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/framework.pb.h"

namespace cinn::frontend::paddle::utils {

cpp::VarDescAPI::Type TransformVarTypePbToCpp(const ::paddle::framework::proto::VarType::Type& type) {
#define SET_TYPE_CASE_ITEM(type__)                  \
  case ::paddle::framework::proto::VarType::type__: \
    return cpp::VarDescAPI::Type::type__;           \
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
      return cpp::VarDescAPI::Type::RAW;
  }
#undef SET_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarTypeCppToPb(const cpp::VarDescAPI::Type& type) {
#define SET_TYPE_CASE_ITEM(type__)                      \
  case cpp::VarDescAPI::Type::type__:                   \
    return ::paddle::framework::proto::VarType::type__; \
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
      return ::paddle::framework::proto::VarType::RAW;
  }
#undef SET_TYPE_CASE_ITEM
}

cpp::VarDescAPI::Type TransformVarDataTypePbToCpp(const ::paddle::framework::proto::VarType::Type& type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)             \
  case ::paddle::framework::proto::VarType::type__: \
    return cpp::VarDescAPI::Type::type__;           \
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
      return cpp::VarDescAPI::Type::RAW;
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarDataTypeCppToPb(const cpp::VarDescAPI::Type& type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                 \
  case cpp::VarDescAPI::Type::type__:                   \
    return ::paddle::framework::proto::VarType::type__; \
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
      return ::paddle::framework::proto::VarType::RAW;
  }
#undef SET_DATA_TYPE_CASE_ITEM
}
}  // namespace cinn::frontend::paddle::utils

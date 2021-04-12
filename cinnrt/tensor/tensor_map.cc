#include "cinnrt/tensor/tensor_map.h"

#include <fstream>
#include <iostream>

#include "cinn/frontend/paddle/compatible_pb.h"
#include "cinn/frontend/paddle/model_parser.h"

using Scope       = cinn::hlir::framework::Scope;
using ProgramDesc = cinn::frontend::paddle::cpp::ProgramDesc;
using Target      = cinn::common::Target;

namespace cinnrt {
namespace tensor {

cinnrt::DType CinnType2DType_(cinn::common::Type type) {
  if (type.is_bool()) return GetDType<bool>();
  if (type.is_int(8)) return GetDType<int8_t>();
  if (type.is_int(16)) return GetDType<int16_t>();
  if (type.is_int(32)) return GetDType<int32_t>();
  if (type.is_int(64)) return GetDType<int64_t>();
  if (type.is_uint(8)) return GetDType<uint8_t>();
  if (type.is_uint(16)) return GetDType<uint16_t>();
  if (type.is_uint(32)) return GetDType<uint32_t>();
  if (type.is_uint(64)) return GetDType<uint64_t>();
  if (type.is_float(32)) return GetDType<float>();
  if (type.is_float(64)) return GetDType<double>();
  if (type.is_string()) return GetDType<std::string>();
  return cinnrt::DType(cinnrt::DType::Kind::Unk);
}

TensorMap *LoadParams(const std::string &path) {
  std::cout << "loading params from: " << path << std::endl;
  TensorMap *map = new TensorMap();
  Scope scope;
  ProgramDesc cpp_prog;
  const Target &target = cinn::common::DefaultHostTarget();

  std::string model_path                              = path + "/__model__";
  paddle::framework::proto::ProgramDesc pb_proto_prog = *cinn::frontend::paddle::LoadProgram(model_path);
  // cinn::frontend::paddle::pb::ProgramDesc pb_prog_desc(&pb_proto_prog);
  // cinn::frontend::paddle::TransformProgramDescAnyToCpp(pb_prog_desc, cpp_prog);
  auto main_block = pb_proto_prog.blocks(0);
  for (auto &var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) continue;
    std::string param_path = path + "/" + var.name();
    std::ifstream param_file(param_path, std::ios::binary);
    switch (var.type().type()) {
      case paddle::framework::proto::VarType_Type_LOD_TENSOR: {
        using CinnTensor = cinn::hlir::framework::Tensor;
        using namespace cinn::utils;
        auto var_name = TransValidVarName(var.name());
        // std::cout << "var name: " << var.name() << " " << var_name << std::endl;
        auto *_var = scope.Var<CinnTensor>(var_name);
        cinn::frontend::paddle::LoadLoDTensor(param_file, _var, target);
        auto tensor     = scope.GetTensor(var_name);
        auto *src_data  = tensor->data<float>();
        auto &cinn_type = tensor->type();
        std::vector<int64_t> shape;
        for (int dim : tensor->shape().data()) shape.push_back(dim);
        auto shape_array = llvm::ArrayRef<int64_t>(shape.data(), shape.size());
        auto dtype       = CinnType2DType_(cinn_type);
        auto *dht        = new DenseHostTensor(TensorShape(shape_array), dtype);
        int num_elements = dht->shape().GetNumElements();
        auto *dst_data   = reinterpret_cast<float *>(dht->raw_data());
        for (int i = 0; i < num_elements; ++i) dst_data[i] = src_data[i];
        (*map)[var.name()] = dht;
        break;
      }
      default:
        std::cout << "unknown weight type" << std::endl;
        break;
    }
  }
  return map;
}

}  // namespace tensor
}  // namespace cinnrt

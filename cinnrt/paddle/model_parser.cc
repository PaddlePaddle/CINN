#include "cinnrt/paddle/model_parser.h"

#include <fstream>
#include <vector>

#include "cinnrt/cinn/string.h"
#include "cinnrt/common/common.h"
#include "cinnrt/common/target.h"
#include "cinnrt/common/type.h"
#include "cinnrt/paddle/compatible_pb.h"
#include "cinnrt/paddle/scope.h"
#include "cinnrt/paddle/tensor.h"

namespace cinnrt::paddle {

int SizeOfType(framework_proto::VarType::Type type) {
  using Type = framework_proto::VarType::Type;
  switch (static_cast<int>(type)) {
#define DO(desc, type)            \
  case Type::VarType_Type_##desc: \
    return sizeof(type);
    DO(BOOL, bool);
    DO(FP16, float);
    DO(FP32, float);
    DO(INT8, int8_t);
    DO(INT16, int16_t);
    DO(INT32, int);
    DO(INT64, int64_t);
#undef DO
    default:
      LOG(FATAL) << "unknown data type " << type;
  }
  return -1;
}

void TensorFromStream(std::istream &is, cinnrt::paddle::_Tensor_ *tensor, const cinnrt::common::Target &target) {
  using Type = framework_proto::VarType::Type;
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK_EQ(version, 0U) << "Only version 0 is supported";
  // read tensor desc
  framework_proto::VarType::TensorDesc desc;
  {
    // int32_t size
    // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char *>(buf.get()), size);
    CHECK(desc.ParseFromArray(buf.get(), size)) << "Cannot parse tensor desc";
  }

  // read tensor
  std::vector<int32_t> dims_vec;
  std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims_vec));
  cinnrt::paddle::Shape dims(dims_vec);
  tensor->Resize(dims);
  void *buf;
  size_t size = tensor->shape().numel() * SizeOfType(desc.data_type());
  // alllocate memory
  if (target.arch == cinnrt::common::Target::Arch::X86) {
    switch (static_cast<int>(desc.data_type())) {
#define SET_TENSOR(desc, type, precision)     \
  case Type::VarType_Type_##desc:             \
    buf = tensor->mutable_data<type>(target); \
    tensor->set_type(precision);              \
    break

      SET_TENSOR(FP32, float, cinnrt::common::Float(32));
      SET_TENSOR(INT8, int8_t, cinnrt::common::Int(8));
      SET_TENSOR(INT16, int16_t, cinnrt::common::Int(16));
      SET_TENSOR(INT32, int32_t, cinnrt::common::Int(32));
      SET_TENSOR(INT64, int64_t, cinnrt::common::Int(64));
#undef SET_TENSOR
      default:
        LOG(FATAL) << "unknown type " << desc.data_type();
    }
    // tensor->set_persistable(true);
    is.read(static_cast<char *>(buf), size);
  } else if (target.arch == cinnrt::common::Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
    if (desc.data_type() != Type::VarType_Type_FP32) LOG(FATAL) << "[CUDA] The type is not fp32!!";
    auto *data = tensor->mutable_data<float>(target);
    tensor->set_type(cinnrt::common::Float(32));
    std::vector<float> temp(tensor->shape().numel());
    // LOG(INFO) <<"[CUDA] The tensor's size is "<< tensor->shape().numel();
    is.read(reinterpret_cast<char *>(temp.data()), size);
    CUDA_CALL(cudaMemcpy(
        reinterpret_cast<void *>(data), temp.data(), tensor->shape().numel() * sizeof(float), cudaMemcpyHostToDevice));
#else
    LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void LoadLoDTensor(std::istream &is, cinnrt::paddle::_Variable *var, const cinnrt::common::Target &target) {
  auto &tensor = std::get<cinnrt::paddle::Tensor>(*var);
  uint32_t version{};
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  VLOG(3) << "model version " << version;

  // Load LoD information
  uint64_t lod_level{};
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));

  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<uint64_t> tmp(size / sizeof(uint64_t));
    is.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(size));
    // lod[i] = tmp;
  }

  TensorFromStream(is, tensor.operator->(), target);
}

void ReadBinaryFile(const std::string &filename, std::string *contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "Cannot open file: " << filename;
  fin.seekg(0, std::ios::end);
  auto size = fin.tellg();
  contents->clear();
  contents->resize(size);
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(const std::string &path, bool program_from_memory) {
  std::unique_ptr<framework_proto::ProgramDesc> main_program(new framework_proto::ProgramDesc);
  if (!program_from_memory) {
    std::string desc_str;
    ReadBinaryFile(path, &desc_str);
    main_program->ParseFromString(desc_str);
  } else {
    main_program->ParseFromString(path);
  }
  return main_program;
}

void LoadParams(const std::string &path) {}

// Load directly to CPU, and latter transfer to other devices.
void LoadParam(const std::string &path, cinnrt::paddle::_Variable *out, const cinnrt::common::Target &target) {
  std::ifstream fin(path, std::ios::binary);
  CHECK(fin.is_open()) << "failed to open file " << path;
  LoadLoDTensor(fin, out, target);
}

bool IsPersistable(const ::cinnrt::paddle::cpp::VarDesc &var) {
  if (var.Persistable() && var.GetType() != ::cinnrt::paddle::cpp::VarDescAPI::Type::FEED_MINIBATCH &&
      var.GetType() != ::cinnrt::paddle::cpp::VarDescAPI::Type::FETCH_LIST &&
      var.GetType() != ::cinnrt::paddle::cpp::VarDescAPI::Type::RAW) {
    return true;
  }
  return false;
}

void LoadCombinedParamsPb(const std::string &path,
                          cinnrt::paddle::Scope *scope,
                          const ::cinnrt::paddle::cpp::ProgramDesc &cpp_prog,
                          bool params_from_memory,
                          const cinnrt::common::Target &target) {
  CHECK(scope);
  auto prog             = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<::cinnrt::paddle::cpp::BlockDesc>(0);

  // Get vars
  std::vector<std::string> paramlist;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<::cinnrt::paddle::cpp::VarDesc>(i);
    if (!IsPersistable(var)) continue;
    paramlist.push_back(var.Name());
  }
  std::sort(paramlist.begin(), paramlist.end());

  // Load vars
  auto load_var_func = [&](std::istream &is) {
    for (size_t i = 0; i < paramlist.size(); ++i) {
      auto *var = scope->Var<cinnrt::paddle::Tensor>(cinnrt::cinn::TransValidVarName(paramlist[i]));
      // Error checking
      CHECK(static_cast<bool>(is)) << "There is a problem with loading model parameters";
      LoadLoDTensor(is, var, target);
    }
    is.peek();
    CHECK(is.eof()) << "You are not allowed to load partial data via"
                    << " LoadCombinedParamsPb, use LoadParam instead.";
  };

  if (params_from_memory) {
    std::stringstream fin(path, std::ios::in | std::ios::binary);
    load_var_func(fin);
  } else {
    std::ifstream fin(path, std::ios::binary);
    CHECK(fin.is_open());
    load_var_func(fin);
  }
}

void LoadModelPb(const std::string &model_dir,
                 const std::string &model_file,
                 const std::string &param_file,
                 cinnrt::paddle::Scope *scope,
                 ::cinnrt::paddle::cpp::ProgramDesc *cpp_prog,
                 bool combined,
                 bool model_from_memory,
                 const cinnrt::common::Target &target) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();
  LOG(INFO) << "model_dir is: " << model_dir;
  LOG(INFO) << "model_file is: " << model_file;
  LOG(INFO) << "param_file is: " << param_file;
  // Load model
  VLOG(4) << "Start load model program...";
  std::string prog_path       = model_dir + "/__model__";
  std::string param_file_temp = param_file;
  if (combined) {
    // prog_path = model_file;
    param_file_temp = model_dir + "/params";
  }
  framework_proto::ProgramDesc pb_proto_prog = *LoadProgram(prog_path, model_from_memory);
  ::cinnrt::paddle::pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to ::cinnrt::paddle::cpp::ProgramDesc
  TransformProgramDescAnyToCpp(pb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  VLOG(4) << "Start load model params...";
  CHECK(!(!combined && model_from_memory)) << "If you want use the model_from_memory,"
                                           << " you should load the combined model using cfg.set_model_buffer "
                                              "interface.";
  if (combined) {
    LoadCombinedParamsPb(param_file_temp, scope, *cpp_prog, model_from_memory, target);
  } else {
    auto main_block = pb_proto_prog.blocks(0);
    for (auto &var : main_block.vars()) {
      if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) continue;

      std::string file_path = model_dir + "/" + var.name();
      VLOG(4) << "reading weight " << var.name();

      std::ifstream file(file_path, std::ios::binary);
      switch (var.type().type()) {
        case framework_proto::VarType_Type_LOD_TENSOR:
          LoadLoDTensor(file, scope->Var<cinnrt::paddle::Tensor>(cinnrt::cinn::TransValidVarName(var.name())), target);
          break;
        default:
          LOG(FATAL) << "unknown weight type";
      }
    }
  }

  VLOG(4) << "Load protobuf model in [" << model_dir << "] successfully";
}

}  // namespace cinnrt::paddle

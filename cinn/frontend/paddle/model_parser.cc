#include "cinn/frontend/paddle/model_parser.h"
#include <fstream>
#include "cinn/common/common.h"
#include "cinn/frontend/paddle/compatible_pb.h"

namespace cinn::frontend::paddle {

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

void TensorFromStream(std::istream &is, hlir::framework::Tensor *tensor) {
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
  std::vector<uint32_t> dims_vec;
  std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims_vec));
  hlir::framework::Shape dims(dims_vec);
  tensor->Resize(dims);
  void *buf;
  size_t size = tensor->shape().numel() * SizeOfType(desc.data_type());
  // alllocate memory
  switch (static_cast<int>(desc.data_type())) {
#define SET_TENSOR(desc, type, precision)                          \
  case Type::VarType_Type_##desc:                                  \
    buf = tensor->mutable_data<type>(common::DefaultHostTarget()); \
    tensor->set_type(precision);                                   \
    break

    SET_TENSOR(FP32, float, Float(32));
    SET_TENSOR(INT8, int8_t, Int(8));
    SET_TENSOR(INT16, int16_t, Int(16));
    SET_TENSOR(INT32, int32_t, Int(32));
    SET_TENSOR(INT64, int64_t, Int(64));
#undef SET_TENSOR
    default:
      LOG(FATAL) << "unknown type " << desc.data_type();
  }
  // tensor->set_persistable(true);

  is.read(static_cast<char *>(buf), size);
}

void LoadLoDTensor(std::istream &is, hlir::framework::Variable *var) {
  auto &tensor = std::get<hlir::framework::Tensor>(*var);
  uint32_t version{};
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  VLOG(3) << "model version " << version;

  // Load LoD information
  uint64_t lod_level{};
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
  /*
  auto &lod = *tensor->mutable_lod();
  lod.resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<uint64_t> tmp(size / sizeof(uint64_t));
    is.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(size));
    lod[i] = tmp;
  }
   */
  TensorFromStream(is, &tensor);
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
void LoadParam(const std::string &path, hlir::framework::Variable *out) {
  std::ifstream fin(path, std::ios::binary);
  CHECK(fin.is_open()) << "failed to open file " << path;
  LoadLoDTensor(fin, out);
}

bool IsPersistable(const cpp::VarDesc &var) {
  if (var.Persistable() && var.GetType() != cpp::VarDescAPI::Type::FEED_MINIBATCH &&
      var.GetType() != cpp::VarDescAPI::Type::FETCH_LIST && var.GetType() != cpp::VarDescAPI::Type::RAW) {
    return true;
  }
  return false;
}

void LoadCombinedParamsPb(const std::string &path,
                          hlir::framework::Scope *scope,
                          const cpp::ProgramDesc &cpp_prog,
                          bool params_from_memory) {
  CHECK(scope);
  auto prog             = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);

  // Get vars
  std::vector<std::string> paramlist;
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (!IsPersistable(var)) continue;
    paramlist.push_back(var.Name());
  }
  std::sort(paramlist.begin(), paramlist.end());

  // Load vars
  auto load_var_func = [&](std::istream &is) {
    for (size_t i = 0; i < paramlist.size(); ++i) {
      auto *var = scope->Var<hlir::framework::Tensor>(paramlist[i]);
      // Error checking
      CHECK(static_cast<bool>(is)) << "There is a problem with loading model parameters";
      LoadLoDTensor(is, var);
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
                 hlir::framework::Scope *scope,
                 cpp::ProgramDesc *cpp_prog,
                 bool combined,
                 bool model_from_memory) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  // Load model
  VLOG(4) << "Start load model program...";
  std::string prog_path = model_dir + "/__model__";
  if (combined) {
    prog_path = model_file;
  }
  framework_proto::ProgramDesc pb_proto_prog = *LoadProgram(prog_path, model_from_memory);
  pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(pb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  VLOG(4) << "Start load model params...";
  CHECK(!(!combined && model_from_memory)) << "If you want use the model_from_memory,"
                                           << " you should load the combined model using cfg.set_model_buffer "
                                              "interface.";
  if (combined) {
    LoadCombinedParamsPb(param_file, scope, *cpp_prog, model_from_memory);
  } else {
    auto main_block = pb_proto_prog.blocks(0);
    for (auto &var : main_block.vars()) {
      if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) continue;

      std::string file_path = model_dir + "/" + var.name();
      VLOG(4) << "reading weight " << var.name();

      std::ifstream file(file_path, std::ios::binary);
      switch (var.type().type()) {
        case framework_proto::VarType_Type_LOD_TENSOR:
          LoadLoDTensor(file, scope->Var<hlir::framework::Tensor>(var.name()));
          break;
        default:
          CHECK(false) << "unknown weight type";
      }
    }
  }

  VLOG(4) << "Load protobuf model in '" << model_dir << "'' successfully";
}

}  // namespace cinn::frontend::paddle

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/paddle/framework.pb.h"
#include "cinn/frontend/paddle/pb/block_desc.h"
#include "cinn/frontend/paddle/pb/op_desc.h"
#include "cinn/frontend/paddle/pb/program_desc.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn::frontend::paddle {
namespace framework_proto = ::paddle::framework::proto;

// Read a model and files of parameters in pb format.
void LoadModelPb(const std::string& model_dir,
                 const std::string& model_file,
                 const std::string& param_file,
                 hlir::framework::Scope* scope,
                 cpp::ProgramDesc* cpp_prog,
                 bool combined          = true,
                 bool model_from_memory = false);

// Read a __model__ file.
std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(const std::string& path, bool program_from_memory = false);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path, hlir::framework::Variable* out);

void LoadCombinedParamsPb(const std::string& path,
                          hlir::framework::Scope* scope,
                          const pb::ProgramDesc& prog,
                          bool params_from_memory = false);

// LoDTensor to ostream
void TensorToStream(std::ostream& os, const hlir::framework::Tensor& tensor);
void TensorFromStream(std::istream& is, hlir::framework::Tensor* tensor);
void ReadBinaryFile(const std::string& filename, std::string* contents);

}  // namespace cinn::frontend::paddle

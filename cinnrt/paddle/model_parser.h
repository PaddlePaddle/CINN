#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "cinnrt/paddle/cpp/program_desc.h"
#include "cinnrt/paddle/framework.pb.h"
#include "cinnrt/paddle/pb/block_desc.h"
#include "cinnrt/paddle/pb/op_desc.h"
#include "cinnrt/paddle/pb/program_desc.h"
#include "cinnrt/paddle/scope.h"
#include "cinnrt/paddle/tensor.h"

namespace cinnrt::paddle {
namespace framework_proto = ::paddle::framework::proto;

// Read a __model__ file.
std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(const std::string& path, bool program_from_memory = false);

void LoadLoDTensor(std::istream& is, cinnrt::paddle::_Variable* var, const cinnrt::common::Target& target);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path, cinnrt::paddle::_Variable* out, const cinnrt::common::Target& target);

void LoadCombinedParamsPb(const std::string& path,
                          cinnrt::paddle::Scope* scope,
                          const ::cinnrt::paddle::pb::ProgramDesc& prog,
                          bool params_from_memory              = false,
                          const cinnrt::common::Target& target = cinnrt::common::DefaultHostTarget());

// LoDTensor to ostream
void TensorToStream(std::ostream& os, const cinnrt::paddle::_Tensor_& tensor);
void TensorFromStream(std::istream& is,
                      cinnrt::paddle::_Tensor_* tensor,
                      const cinnrt::common::Target& target = cinnrt::common::DefaultHostTarget());
void ReadBinaryFile(const std::string& filename, std::string* contents);

}  // namespace cinnrt::paddle

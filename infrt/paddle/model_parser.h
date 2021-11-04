#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "infrt/paddle/framework.pb.h"
#include "infrt/paddle/pb/block_desc.h"
#include "infrt/paddle/pb/op_desc.h"
#include "infrt/paddle/pb/program_desc.h"
#include "infrt/paddle/scope.h"
#include "infrt/paddle/tensor.h"

namespace infrt::paddle {
namespace framework_proto = ::paddle::framework::proto;

// Read a __model__ file.
std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(const std::string& path, bool program_from_memory = false);

void LoadLoDTensor(std::istream& is, infrt::paddle::_Variable* var, const infrt::common::Target& target);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path, infrt::paddle::_Variable* out, const infrt::common::Target& target);

// LoDTensor to ostream
void TensorToStream(std::ostream& os, const infrt::paddle::_Tensor_& tensor);
void TensorFromStream(std::istream& is,
                      infrt::paddle::_Tensor_* tensor,
                      const infrt::common::Target& target = infrt::common::DefaultHostTarget());
void ReadBinaryFile(const std::string& filename, std::string* contents);

}  // namespace infrt::paddle

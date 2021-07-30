#pragma once
#include "cinn/frontend/paddle/cpp/block_desc.h"
#include "cinn/frontend/paddle/cpp/desc_api.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/paddle/cpp/var_desc.h"

namespace cinnrt::paddle {

/// Transform an VarDesc from VarDescType to cpp format.
template <typename VarDescType>
void TransformVarDescAnyToCpp(const VarDescType& any_desc, ::cinn::frontend::paddle::cpp::VarDesc* cpp_desc);

/// Transform an VarDesc from cpp to VarDescType format.
template <typename VarDescType>
void TransformVarDescCppToAny(const ::cinn::frontend::paddle::cpp::VarDesc& cpp_desc, VarDescType* any_desc);

/// Transform an OpDesc from OpDescType to cpp format.
template <typename OpDescType>
void TransformOpDescAnyToCpp(const OpDescType& any_desc, ::cinn::frontend::paddle::cpp::OpDesc* cpp_desc);

/// Transform an OpDesc from cpp to OpDescType format.
template <typename OpDescType>
void TransformOpDescCppToAny(const ::cinn::frontend::paddle::cpp::OpDesc& cpp_desc, OpDescType* any_desc);

/// Transform an BlockDesc from BlockDescType to cpp format.
template <typename BlockDescType>
void TransformBlockDescAnyToCpp(const BlockDescType& any_desc, ::cinn::frontend::paddle::cpp::BlockDesc* cpp_desc);

/// Transform an BlockDesc from cpp to BlockDescType format.
template <typename BlockDescType>
void TransformBlockDescCppToAny(const ::cinn::frontend::paddle::cpp::BlockDesc& cpp_desc, BlockDescType* any_desc);

/// Transform an ProgramDesc from ProgramDescType to cpp format.
template <typename ProgramDescType>
void TransformProgramDescAnyToCpp(const ProgramDescType& any_desc,
                                  ::cinn::frontend::paddle::cpp::ProgramDesc* cpp_desc);

/// Transform an ProgramDesc from cpp to ProgramDescType format.
template <typename ProgramDescType>
void TransformProgramDescCppToAny(const ::cinn::frontend::paddle::cpp::ProgramDesc& cpp_desc,
                                  ProgramDescType* any_desc);

}  // namespace cinnrt::paddle

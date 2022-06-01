// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/backends/llvm/llvm_util.h"

#include <glog/logging.h>
#include <llvm/Support/Alignment.h>

#include <atomic>
#include <mutex>  //NOLINT

namespace cinn {
namespace backends {

llvm::Type *CinnTypeToLLVMType(common::Type type, llvm::Module *m, bool is_vec) {
  llvm::Type *ir_type = nullptr;
  if (type.is_cpp_const()) {
    // TODO(fc500110) support it latter.
  }

  llvm::Type *v   = llvm::Type::getVoidTy(m->getContext());
  llvm::Type *i1  = llvm::Type::getInt1Ty(m->getContext());
  llvm::Type *i8  = llvm::Type::getInt8Ty(m->getContext());
  llvm::Type *u8  = llvm::Type::getInt8Ty(m->getContext());
  llvm::Type *i32 = llvm::Type::getInt32Ty(m->getContext());
  llvm::Type *i64 = llvm::Type::getInt64Ty(m->getContext());
  llvm::Type *u32 = llvm::Type::getInt32Ty(m->getContext());
  llvm::Type *f32 = llvm::Type::getFloatTy(m->getContext());
  llvm::Type *f64 = llvm::Type::getDoubleTy(m->getContext());
  if (type.is_void() && type.is_cpp_handle()) {
    return llvm::PointerType::getUnqual(i8);
  }
  if (type.is_void() && type.is_cpp_handle2()) {
    return llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(i8));
  }

  if (type.is_int(8)) {
    ir_type = i8;
  } else if (type.is_int(32)) {
    ir_type = i32;
  } else if (type.is_int(64)) {
    ir_type = i64;
  } else if (type.is_bool()) {
    ir_type = i1;
  } else if (type.is_float(32)) {
    ir_type = f32;
  } else if (type.is_float(64)) {
    ir_type = f64;
  } else if (type.is_void()) {
    ir_type = v;
  } else if (type.is_customized_type()) {
    CHECK(!type.customized_type().empty());
    ir_type = m->getTypeByName("struct." + type.customized_type());
  }
  CHECK(ir_type) << "LLVM can't convert type: " << type;

  // C array / vector.
  if (type.lanes() > 1) {
    if (is_vec) {
      ir_type = llvm::FixedVectorType::get(ir_type, type.lanes());
    } else {
      ir_type = llvm::ArrayType::get(ir_type, type.lanes());
    }
  }

  if (type.is_cpp_handle()) {
    ir_type = llvm::PointerType::getUnqual(ir_type);
  }

  if (type.is_cpp_handle2()) {
    ir_type = llvm::PointerType::getUnqual(ir_type);
    ir_type = llvm::PointerType::getUnqual(ir_type);
  }

  return ir_type;
}

#define __(ty__)                                           \
  template <>                                              \
  llvm::Type *llvm_type_of<ty__>(llvm::Module * m) {       \
    return CinnTypeToLLVMType(common::type_of<ty__>(), m); \
  }

__(int8_t)
__(int32_t)
__(int64_t)
__(float)
__(double)
__(cinn_buffer_t)
__(cinn_buffer_t *)
__(cinn_pod_value_t *)
__(cinn_pod_value_t)
__(void *)
__(void **)

#undef __

}  // namespace backends
}  // namespace cinn

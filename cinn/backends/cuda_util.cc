#include "cinn/backends/cuda_util.h"

#include <glog/logging.h>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"

namespace cinn {
namespace backends {

std::string cuda_thread_axis_name(int level) {
  switch (level) {
    case 0:
      return "threadIdx.x";
      break;
    case 1:
      return "threadIdx.y";
      break;
    case 2:
      return "threadIdx.z";
      break;
  }
  return "";
}

std::string cuda_block_axis_name(int level) {
  switch (level) {
    case 0:
      return "blockIdx.x";
      break;
    case 1:
      return "blockIdx.y";
      break;
    case 2:
      return "blockIdx.z";
      break;
  }
  return "";
}

}  // namespace backends
}  // namespace cinn

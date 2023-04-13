

#include "cinn/ir/ir_verify.h"

#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir.h"
#include <iostream>
#include "cinn/ir/tensor.h"
#include "cinn/lang/placeholder.h"

#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_util.h"

#include "cinn/backends/nvrtc/nvrtc_util.h"

#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/hlir/framework/pass.h"

#include "cinn/frontend/net_builder.h"

#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>

#include "cinn/lang/placeholder.h"

#include "cinn/hlir/framework/visualize_helper.h"
#include "cinn/hlir/framework/graph.h"

#include <cmath>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

#pragma once

namespace cinn {
namespace ir {

using GroupPtr = std::shared_ptr<hlir::framework::Graph::Group>;


enum ReduceType {  
  kContiguous = 0,
  
  kNoConguous = 1,
};

struct CodeGenOption
{
  ReduceType reduce_type;
  
  int flatten_block;
  int reduce_block;
  int num_warp;
  int num_thread_per_warp;
};

ir::LoweredFunc process_warp_reduce(  hlir::framework::Graph* graph, CodeGenOption gen_opt, 
      const std::vector<std::string>& vec_input,  const std::vector<std::string>& vec_output);


} //namespace ir
} // namespce cinn
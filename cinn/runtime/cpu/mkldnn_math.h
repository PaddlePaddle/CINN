#pragma once
#include "cinn/runtime/cinn_runtime.h"

#ifdef CINN_WITH_MKLDNN
#include "mkldnn.hpp"
#endif

// define some C APIs
extern "C" {

void cinn_cpu_mkldnn_conv2d_nchw_fp32(int batch_size,
                                      int c_in,
                                      int input_h,
                                      int input_w,
                                      int c_out,
                                      int group,
                                      int filter_h,
                                      int filter_w,
                                      int pad_h,
                                      int pad_w,
                                      int stride_h,
                                      int stride_w,
                                      int dilation_h,
                                      int dilation_w,
                                      cinn_buffer_t* inputs,
                                      cinn_buffer_t* weights,
                                      cinn_buffer_t* out);

}  // extern "C"

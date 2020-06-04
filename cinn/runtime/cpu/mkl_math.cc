#include "mkl_math.h"

#include <glog/logging.h>
#include <mkl.h>
#include <mkl_vml_functions.h>

#include <cmath>

void cinn_mkl_tanh_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vsTanh(x->num_elements(), reinterpret_cast<float *>(x->host_memory), reinterpret_cast<float *>(out->host_memory));
}
void cinn_mkl_tanh_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdTanh(x->num_elements(), reinterpret_cast<double *>(x->host_memory), reinterpret_cast<double *>(out->host_memory));
}
void cinn_mkl_exp_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdExp(x->num_elements(), reinterpret_cast<double *>(x->host_memory), reinterpret_cast<double *>(out->host_memory));
}

/*
void cinn_mkl_cos_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vsCosh(x->num_elements(), reinterpret_cast<float *>(x->host_memory), reinterpret_cast<float *>(out->host_memory));
}
void cinn_mkl_cos_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdCosh(x->num_elements(), reinterpret_cast<double *>(x->host_memory), reinterpret_cast<double *>(out->host_memory));
}
*/

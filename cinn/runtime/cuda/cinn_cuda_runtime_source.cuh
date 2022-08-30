/**
 * \file This file contains all the intrinsics available to be used in CUDA code generated by CodeGen.
 */

#define CINN_FLT_MAX 3.402823e+38f
#define CINN_FLT_MIN -3.402823e+38f

#define FN(x) cinn_nvgpu_##x##_fp32
// NOTE Due to function override, we don't need to use type (such as '_fp32') as the suffix of function's name.
__device__ inline float FN(sin)(float x) { return sin(x); }
__device__ inline float FN(cos)(float x) { return cos(x); }
__device__ inline float FN(cosh)(float x) { return cosh(x); }
__device__ inline float FN(tanh)(float x) { return tanh(x); }

__device__ inline float FN(asin)(float x) { return asin(x); }
__device__ inline float FN(acos)(float x) { return acos(x); }
__device__ inline float FN(acosh)(float x) { return acosh(x); }
__device__ inline float FN(atanh)(float x) { return atanh(x); }

__device__ inline float FN(ceil)(float x) { return ceil(x); }
__device__ inline float FN(round)(float x) { return round(x); }
__device__ inline float FN(trunc)(float x) { return trunc(x); }
__device__ inline float FN(abs)(float x) { return abs(x); }
__device__ inline float FN(floor)(float x) { return floor(x); }
__device__ inline float FN(log)(float x) { return log(x); }
__device__ inline float FN(log2)(float x) { return log2(x); }
__device__ inline float FN(log10)(float x) { return log10(x); }
__device__ inline float FN(exp)(float x) { return exp(x); }
__device__ inline float FN(erf)(float x) { return erf(x); }
__device__ inline float FN(sigmoid)(float x) { return 1. / (1 + exp(-x)); }
__device__ inline float FN(sqrt)(float x) { return sqrt(x); }
__device__ inline float FN(rsqrt)(float x) { return rsqrt(x); }

__device__ inline bool FN(isfinite)(float x) { return isfinite(x); }
__device__ inline bool FN(isinf)(float x) { return isinf(x); }
__device__ inline bool FN(isnan)(float x) { return isnan(x); }

__device__ inline float FN(max)(float a, float b) { return max(a, b); }
__device__ inline float FN(min)(float a, float b) { return min(a, b); }

#undef FN

__device__ inline float cinn_sum(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod(const float left, const float right) { return left * right; }
__device__ inline float cinn_max(const float left, const float right) { return max(left, right); }
__device__ inline float cinn_min(const float left, const float right) { return min(left, right); }
__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }

#define cinn_warp_shuffle_internal_kernel(TYPE, value, op)                  \
  TYPE tmp_val      = value;                                                \
  unsigned int mask = __activemask();                                       \
  tmp_val           = op(tmp_val, __shfl_down_sync(mask, tmp_val, 16, 32)); \
  tmp_val           = op(tmp_val, __shfl_down_sync(mask, tmp_val, 8, 32));  \
  tmp_val           = op(tmp_val, __shfl_down_sync(mask, tmp_val, 4, 32));  \
  tmp_val           = op(tmp_val, __shfl_down_sync(mask, tmp_val, 2, 32));  \
  tmp_val           = op(tmp_val, __shfl_down_sync(mask, tmp_val, 1, 32));  \
  tmp_val           = __shfl_sync(mask, tmp_val, 0, 32);                    \
  return tmp_val;

__device__ inline float cinn_warp_shuffle_sum_internal(const float value) {
  cinn_warp_shuffle_internal_kernel(float, value, cinn_sum);
}
__device__ inline float cinn_warp_shuffle_prod_internal(const float value) {
  cinn_warp_shuffle_internal_kernel(float, value, cinn_prod);
}
__device__ inline float cinn_warp_shuffle_max_internal(const float value) {
  cinn_warp_shuffle_internal_kernel(float, value, cinn_max);
}
__device__ inline float cinn_warp_shuffle_min_internal(const float value) {
  cinn_warp_shuffle_internal_kernel(float, value, cinn_min);
}
__device__ inline bool cinn_warp_shuffle_all_internal(const bool value) {
  cinn_warp_shuffle_internal_kernel(bool, value, cinn_all);
}
__device__ inline bool cinn_warp_shuffle_any_internal(const bool value) {
  cinn_warp_shuffle_internal_kernel(bool, value, cinn_any);
}

#undef cinn_warp_shuffle_internal_kernel

__device__ inline float cinn_warp_reduce_max(const float *buf, int offset, int extend) {
  float tmp_val = CINN_FLT_MIN;
  for (int i = threadIdx.x; i < extend; i += 32) {
    tmp_val = max(tmp_val, buf[offset + i]);
  }
  return cinn_warp_shuffle_max_internal(tmp_val);
}

__device__ inline float cinn_warp_reduce_sum(const float *buf, int offset, int extend) {
  float tmp_val = 0.0f;
  for (int i = threadIdx.x; i < extend; i += 32) {
    tmp_val += buf[offset + i];
  }
  return cinn_warp_shuffle_sum_internal(tmp_val);
}

__device__ inline float cinn_warp_reduce_avg(const float *buf, int offset, int extend) {
  return cinn_warp_reduce_sum(buf, offset, extend) / extend;
}

#define cinn_block_reduce_internal_kernel(TYPE, value, init_value, cinn_warp_shuffle_internal) \
  int warp_id = threadIdx.x / 32;                                                              \
  __shared__ TYPE tmp[32];                                                                     \
  if (warp_id == 0) {                                                                          \
    tmp[threadIdx.x] = init_value;                                                             \
  }                                                                                            \
  TYPE tmp_val = cinn_warp_shuffle_internal(value);                                            \
  if (blockDim.x <= 32) {                                                                      \
    return tmp_val;                                                                            \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (threadIdx.x % 32 == 0) {                                                                 \
    tmp[warp_id] = tmp_val;                                                                    \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (warp_id == 0) {                                                                          \
    tmp_val = tmp[threadIdx.x];                                                                \
    tmp_val = cinn_warp_shuffle_internal(tmp_val);                                             \
    if (threadIdx.x == 0) {                                                                    \
      tmp[0] = tmp_val;                                                                        \
    }                                                                                          \
  }                                                                                            \
  __syncthreads();                                                                             \
  return tmp[0];

// block reduce sum internal
__device__ inline float cinn_block_reduce_sum_internal(const float value) {
  cinn_block_reduce_internal_kernel(float, value, 0.0f, cinn_warp_shuffle_sum_internal);
}
// block reduce prod internal
__device__ inline float cinn_block_reduce_prod_internal(const float value) {
  cinn_block_reduce_internal_kernel(float, value, 1.0f, cinn_warp_shuffle_prod_internal);
}
// block reduce max internal
__device__ inline float cinn_block_reduce_max_internal(const float value) {
  cinn_block_reduce_internal_kernel(float, value, CINN_FLT_MIN, cinn_warp_shuffle_max_internal);
}
// block reduce min internal
__device__ inline float cinn_block_reduce_min_internal(const float value) {
  cinn_block_reduce_internal_kernel(float, value, CINN_FLT_MAX, cinn_warp_shuffle_min_internal);
}
// block reduce all internal
__device__ inline bool cinn_block_reduce_all_internal(const bool value) {
  cinn_block_reduce_internal_kernel(bool, value, true, cinn_warp_shuffle_all_internal);
}
// block reduce any internal
__device__ inline bool cinn_block_reduce_any_internal(const bool value) {
  cinn_block_reduce_internal_kernel(bool, value, false, cinn_warp_shuffle_any_internal);
}

#undef cinn_block_reduce_internal_kernel

// block reduce sum
__device__ inline float cinn_block_reduce_sum(const float *buf, int offset, int extend) {
  float tmp_val = 0.0f;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val += buf[offset + i];
  }
  return cinn_block_reduce_sum_internal(tmp_val);
}
// block reduce prod
__device__ inline float cinn_block_reduce_prod(const float *buf, int offset, int extend) {
  float tmp_val = 1.0f;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val *= buf[offset + i];
  }
  return cinn_block_reduce_prod_internal(tmp_val);
}
// block reduce max
__device__ inline float cinn_block_reduce_max(const float *buf, int offset, int extend) {
  float tmp_val = CINN_FLT_MIN;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val = max(tmp_val, buf[offset + i]);
  }
  return cinn_block_reduce_max_internal(tmp_val);
}
// block reduce min
__device__ inline float cinn_block_reduce_min(const float *buf, int offset, int extend) {
  float tmp_val = CINN_FLT_MAX;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val = min(tmp_val, buf[offset + i]);
  }
  return cinn_block_reduce_min_internal(tmp_val);
}
// block reduce all
__device__ inline bool cinn_block_reduce_all(const bool *buf, int offset, int extend) {
  bool tmp_val = true;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val = tmp_val && buf[offset + i];
  }
  return cinn_block_reduce_all_internal(tmp_val);
}
// block reduce any
__device__ inline bool cinn_block_reduce_any(const bool *buf, int offset, int extend) {
  bool tmp_val = false;
  for (int i = threadIdx.x; i < extend; i += blockDim.x) {
    tmp_val = tmp_val || buf[offset + i];
  }
  return cinn_block_reduce_any_internal(tmp_val);
}

#define __cinn_cuda_find_kernel(buf, size, num) \
  do {                                          \
    for (int i = size - 1; i >= 0; --i) {       \
      if (buf[i] == num) return i;              \
    }                                           \
    return -1;                                  \
  } while (0)

__device__ inline int cinn_cuda_find_int(const int *buf, int size, int num) { __cinn_cuda_find_kernel(buf, size, num); }

__device__ inline int cinn_cuda_find_float(const float *buf, int size, float num) {
  __cinn_cuda_find_kernel(buf, size, num);
}

#undef __cinn_cuda_find_kernel

#define __cinn_cuda_find_from_kernel(buf, size, num, begin) \
  do {                                                      \
    for (int i = begin; i < size; ++i) {                    \
      if (buf[i] == num) return i;                          \
    }                                                       \
    return -1;                                              \
  } while (0)

__device__ inline int cinn_cuda_find_int_from(const int *buf, int size, int num, int begin) {
  __cinn_cuda_find_from_kernel(buf, size, num, begin);
}

__device__ inline int cinn_cuda_find_float_from(const float *buf, int size, float num, int begin) {
  __cinn_cuda_find_from_kernel(buf, size, num, begin);
}

#undef __cinn_cuda_find_from_kernel

__device__ inline float cinn_cuda_index_add(const float x,
                                            const int axis_indice,
                                            const float *__restrict__ y,
                                            const int offset,
                                            const int stride,
                                            const int *__restrict__ index,
                                            const int index_size) {
  float res = x;
  int idx   = -1;
  do {
    idx = cinn_cuda_find_int_from(index, index_size, axis_indice, idx + 1);
    if (idx >= 0) {
      res += y[offset + idx * stride];
    }
  } while (idx != -1);
  return res;
}

#define block_shuffle_kernel(TYPE, name, op, init_value)                               \
  __device__ inline TYPE block_shuffle_##name(const TYPE *buf, int line, int stride) { \
    TYPE val = init_value;                                                             \
    for (int idx = threadIdx.x; idx < line; idx += stride) {                           \
      val = op(val, buf[idx]);                                                         \
    }                                                                                  \
    return val;                                                                        \
  }

block_shuffle_kernel(float, sum, cinn_sum, 0.0f);
block_shuffle_kernel(float, prod, cinn_prod, 1.0f);
block_shuffle_kernel(float, max, cinn_max, CINN_FLT_MIN);
block_shuffle_kernel(float, min, cinn_min, CINN_FLT_MAX);
block_shuffle_kernel(bool, all, cinn_all, true);
block_shuffle_kernel(bool, any, cinn_any, false);

#undef block_shuffle_kernel
#undef CINN_FLT_MIN
#undef CINN_FLT_MAX

/**
 * \file This file contains all the intrinsics available to be used in CUDA code generated by CodeGen.
 */

#ifdef __cplusplus

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(x)); }
__device__ inline double sigmoid(double x) { return 1.0 / (1.0 + exp(x)); }

__device__ inline int left_shift(int a, int b) { return a << b; }
__device__ inline int64_t left_shift(int64_t a, int64_t b) { return a << b; }

__device__ inline int right_shift(int a, int b) { return a >> b; }
__device__ inline int64_t right_shift(int64_t a, int64_t b) { return a >> b; }

__device__ inline int bitwise_and(int a, int b) { return a & b; }
__device__ inline int64_t bitwise_and(int64_t a, int64_t b) { return a & b; }

__device__ inline int bitwise_or(int a, int b) { return a | b; }
__device__ inline int64_t bitwise_or(int64_t a, int64_t b) { return a | b; }

__device__ inline int bitwise_xor(int a, int b) { return a ^ b; }
__device__ inline int64_t bitwise_xor(int64_t a, int64_t b) { return a ^ b; }

__device__ inline int bitwise_not(int a) { return ~a; }
__device__ inline int64_t bitwise_not(int64_t a) { return ~a; }

__device__ inline int logical_right_shift(int a, int b) { return ((unsigned int)a >> b); }
__device__ inline int64_t logical_right_shift(int64_t a, int64_t b) { return ((uint64_t)a >> b); }

__device__ inline int clz(int a) { return __clz(a); }
__device__ inline int64_t clz(int64_t a) { return __clzll(a); }

__device__ inline int popc(int a) { return __popc(a); }
__device__ inline int64_t popc(int64_t a) { return __popcll(a); }

#endif  // __cplusplus

extern "C" {
// *************************************************************** //
// reduce operator, need `--expt-relaxed-constexpr` option to call std function in device kernel
#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...)          \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__)         \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__)        \
  MACRO(max_fp32, -3.40282e+38, float, ##__VA_ARGS__) \
  MACRO(min_fp32, 3.40282e+38, float, ##__VA_ARGS__)

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return max(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return min(left, right); }

#ifdef CINN_CUDA_FP16

#define EXPAND_REDUCE_FP16_MACRO(MACRO, ...)                                           \
  MACRO(sum_fp16, float16(0.0), float16, ##__VA_ARGS__)                                \
  MACRO(prod_fp16, float16(1.0), float16, ##__VA_ARGS__)                               \
  MACRO(max_fp16, cinn::common::raw_uint16_to_float16(0xfbff), float16, ##__VA_ARGS__) \
  MACRO(min_fp16, cinn::common::raw_uint16_to_float16(0x7bff), float16, ##__VA_ARGS__)

__device__ inline float16 cinn_sum_fp16(const float16 left, const float16 right) { return left + right; }
__device__ inline float16 cinn_prod_fp16(const float16 left, const float16 right) { return left * right; }
__device__ inline float16 cinn_max_fp16(const float16 left, const float16 right) { return max(left, right); }
__device__ inline float16 cinn_min_fp16(const float16 left, const float16 right) { return min(left, right); }
#endif

#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...) \
  MACRO(all, true, bool, ##__VA_ARGS__)      \
  MACRO(any, false, bool, ##__VA_ARGS__)

__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }

#define CINN_SHUFFLE_FUNCTION(offset, op, init)                  \
  shfl_res = __shfl_down_sync(mask, tmp_val, offset, 32);        \
  shfl_res = threadIdx.x % 32 + offset < lane ? shfl_res : init; \
  tmp_val  = op(tmp_val, shfl_res);

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                \
  __device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
    DTYPE tmp_val     = value, shfl_res;                                                  \
    unsigned int mask = __activemask();                                                   \
    unsigned int lane = __popc(mask);                                                     \
    CINN_SHUFFLE_FUNCTION(16, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE))                   \
    CINN_SHUFFLE_FUNCTION(8, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE))                    \
    CINN_SHUFFLE_FUNCTION(4, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE))                    \
    CINN_SHUFFLE_FUNCTION(2, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE))                    \
    CINN_SHUFFLE_FUNCTION(1, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE))                    \
    tmp_val = __shfl_sync(mask, tmp_val, 0, 32);                                          \
    return tmp_val;                                                                       \
  }

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
#endif

#undef CINN_WARP_SHUFFLE_INTERNAL_IMPL

#define CINN_WARP_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                     \
  __device__ inline DTYPE cinn_warp_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
    DTYPE tmp_val = DTYPE(INITIAL_VALUE);                                                            \
    for (int i = threadIdx.x; i < extend; i += 32) {                                                 \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]);                                        \
    }                                                                                                \
    return cinn_warp_shuffle_##REDUCE_TYPE##_internal(tmp_val);                                      \
  }

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_REDUCE_IMPL)
#endif

#undef CINN_WARP_REDUCE_IMPL

__device__ inline float cinn_warp_reduce_avg_fp32(const float *buf, int offset, int extend) {
  return cinn_warp_reduce_sum_fp32(buf, offset, extend) / extend;
}

#define CINN_BLOCK_REDUCE_INTERNAL_IMPL(TYPE, value, init_value, cinn_warp_shuffle_internal) \
  int warp_id = threadIdx.x / 32;                                                            \
  __shared__ TYPE tmp[32];                                                                   \
  if (warp_id == 0) {                                                                        \
    tmp[threadIdx.x] = init_value;                                                           \
  }                                                                                          \
  TYPE tmp_val = cinn_warp_shuffle_internal(value);                                          \
  if (blockDim.x <= 32) {                                                                    \
    return tmp_val;                                                                          \
  }                                                                                          \
  __syncthreads();                                                                           \
  if (threadIdx.x % 32 == 0) {                                                               \
    tmp[warp_id] = tmp_val;                                                                  \
  }                                                                                          \
  __syncthreads();                                                                           \
  if (warp_id == 0) {                                                                        \
    tmp_val = tmp[threadIdx.x];                                                              \
    tmp_val = cinn_warp_shuffle_internal(tmp_val);                                           \
    if (threadIdx.x == 0) {                                                                  \
      tmp[0] = tmp_val;                                                                      \
    }                                                                                        \
  }                                                                                          \
  __syncthreads();                                                                           \
  return tmp[0];

#define CINN_BLOCK_REDUCE_INTERNAL_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                          \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE##_internal(const DTYPE value) {                            \
    CINN_BLOCK_REDUCE_INTERNAL_IMPL(DTYPE, value, DTYPE(INITIAL_VALUE), cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
#endif

#undef CINN_BLOCK_REDUCE_INTERNAL_IMPL
#undef CINN_BLOCK_REDUCE_INTERNAL_MACRO

#define CINN_BLOCK_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                     \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
    DTYPE tmp_val = DTYPE(INITIAL_VALUE);                                                             \
    for (int i = threadIdx.x; i < extend; i += blockDim.x) {                                          \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]);                                         \
    }                                                                                                 \
    return cinn_block_reduce_##REDUCE_TYPE##_internal(tmp_val);                                       \
  }

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_IMPL)
#endif

#undef CINN_BLOCK_REDUCE_IMPL

#define BLOCK_SHUFFLE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                   \
  __device__ inline DTYPE block_shuffle_##REDUCE_TYPE(const DTYPE *buf, int line, int stride) { \
    DTYPE val = DTYPE(INITIAL_VALUE);                                                           \
    for (int idx = threadIdx.x; idx < line; idx += stride) {                                    \
      val = cinn_##REDUCE_TYPE(val, buf[idx]);                                                  \
    }                                                                                           \
    return val;                                                                                 \
  }

EXPAND_REDUCE_FP32_MACRO(BLOCK_SHUFFLE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(BLOCK_SHUFFLE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(BLOCK_SHUFFLE_IMPL)
#endif

#undef BLOCK_SHUFFLE_IMPL

#undef EXPAND_REDUCE_FP32_MACRO
#undef EXPAND_REDUCE_BOOL_MACRO

#ifdef CINN_CUDA_FP16
#undef EXPAND_REDUCE_FP16_MACRO
#endif

// *************************************************************** //
// other function
#define __cinn_cuda_find_kernel(buf, size, num, begin, stride)           \
  do {                                                                   \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) { \
      if (buf[i] == num) return (i - begin) / stride;                    \
    }                                                                    \
    return -1;                                                           \
  } while (0)

__device__ inline int cinn_cuda_find_int(const int *buf, int size, int num) {
  __cinn_cuda_find_kernel(buf, size, num, 0, 1);
}

__device__ inline int cinn_cuda_find_float(const float *buf, int size, float num) {
  __cinn_cuda_find_kernel(buf, size, num, 0, 1);
}

__device__ inline int cinn_cuda_find_int_nd(const int *buf, int size, int num, int begin, int stride) {
  __cinn_cuda_find_kernel(buf, size, num, begin, stride);
}

__device__ inline int cinn_cuda_find_float_nd(const float *buf, int size, float num, int begin, int stride) {
  __cinn_cuda_find_kernel(buf, size, num, begin, stride);
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

#define __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride)          \
  do {                                                                     \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] < num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  } while (0)

__device__ inline int cinn_cuda_lt_num_float(
    const float *buf, const int size, const float num, const int offset, const int stride) {
  __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride);
}

__device__ inline int cinn_cuda_lt_num_int(
    const int *buf, const int size, const int num, const int offset, const int stride) {
  __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride);
}

#undef __cinn_cuda_lt_num_kernel

#define __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride)          \
  do {                                                                     \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] > num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  } while (0)

__device__ inline int cinn_cuda_gt_num_float(
    const float *buf, const int size, const float num, const int offset, const int stride) {
  __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride);
}

__device__ inline int cinn_cuda_gt_num_int(
    const int *buf, const int size, const int num, const int offset, const int stride) {
  __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride);
}

#undef __cinn_cuda_gt_num_kernel

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

// *************************************************************** //
// end of macro undef
#undef FN_FP32
#undef FN_FP64
#undef FN_INT32
#undef FN_INT64

#ifdef CINN_CUDA_FP16
#undef FN_FP16
#endif
}

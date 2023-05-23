/**
 * \file This file contains all the intrinsics available to be used in CUDA code generated by CodeGen.
 */

extern "C" {

#define CINN_INT32_MAX 2147483647
#define CINN_INT32_MIN -2147483648

// *************************************************************** //
// bool unary and binary operator
#define FN_BOOL(func) cinn_nvgpu_##func##_bool
__device__ inline bool FN_BOOL(bitwise_and)(bool a, bool b) { return a & b; }
__device__ inline bool FN_BOOL(bitwise_or)(bool a, bool b) { return a | b; }
__device__ inline bool FN_BOOL(bitwise_xor)(bool a, bool b) { return a ^ b; }
__device__ inline bool FN_BOOL(bitwise_not)(bool a) { return !a; }

// *************************************************************** //
// uint8 unary and binary operator
#define FN_UINT8(func) cinn_nvgpu_##func##_uint8
__device__ inline uint8_t FN_UINT8(bitwise_and)(uint8_t a, uint8_t b) { return a & b; }
__device__ inline uint8_t FN_UINT8(bitwise_or)(uint8_t a, uint8_t b) { return a | b; }
__device__ inline uint8_t FN_UINT8(bitwise_xor)(uint8_t a, uint8_t b) { return a ^ b; }
__device__ inline uint8_t FN_UINT8(bitwise_not)(uint8_t a) { return ~a; }

// *************************************************************** //
// int8 unary and binary operator
#define FN_INT8(func) cinn_nvgpu_##func##_int8
__device__ inline int8_t FN_INT8(bitwise_and)(int8_t a, int8_t b) { return a & b; }
__device__ inline int8_t FN_INT8(bitwise_or)(int8_t a, int8_t b) { return a | b; }
__device__ inline int8_t FN_INT8(bitwise_xor)(int8_t a, int8_t b) { return a ^ b; }
__device__ inline int8_t FN_INT8(bitwise_not)(int8_t a) { return ~a; }

// *************************************************************** //
// int16 unary and binary operator
#define FN_INT16(func) cinn_nvgpu_##func##_int16
__device__ inline int16_t FN_INT16(bitwise_and)(int16_t a, int16_t b) { return a & b; }
__device__ inline int16_t FN_INT16(bitwise_or)(int16_t a, int16_t b) { return a | b; }
__device__ inline int16_t FN_INT16(bitwise_xor)(int16_t a, int16_t b) { return a ^ b; }
__device__ inline int16_t FN_INT16(bitwise_not)(int16_t a) { return ~a; }

// *************************************************************** //
// float32 unary and binary operator
#define FN_FP32(func) cinn_nvgpu_##func##_fp32
// NOTE Due to function override, we don't need to use type (such as '_fp32') as the suffix of function's name.
__device__ inline float FN_FP32(sin)(float x) { return sin(x); }
__device__ inline float FN_FP32(cos)(float x) { return cos(x); }
__device__ inline float FN_FP32(tan)(float x) { return tan(x); }
__device__ inline float FN_FP32(sinh)(float x) { return sinh(x); }
__device__ inline float FN_FP32(cosh)(float x) { return cosh(x); }
__device__ inline float FN_FP32(tanh)(float x) { return tanh(x); }

__device__ inline float FN_FP32(asin)(float x) { return asin(x); }
__device__ inline float FN_FP32(acos)(float x) { return acos(x); }
__device__ inline float FN_FP32(atan)(float x) { return atan(x); }
__device__ inline float FN_FP32(asinh)(float x) { return asinh(x); }
__device__ inline float FN_FP32(acosh)(float x) { return acosh(x); }
__device__ inline float FN_FP32(atanh)(float x) { return atanh(x); }

__device__ inline float FN_FP32(ceil)(float x) { return ceil(x); }
__device__ inline float FN_FP32(round)(float x) { return round(x); }
__device__ inline float FN_FP32(trunc)(float x) { return trunc(x); }
__device__ inline float FN_FP32(abs)(float x) { return abs(x); }
__device__ inline float FN_FP32(floor)(float x) { return floor(x); }
__device__ inline float FN_FP32(log)(float x) { return log(x); }
__device__ inline float FN_FP32(log2)(float x) { return log2(x); }
__device__ inline float FN_FP32(log10)(float x) { return log10(x); }
__device__ inline float FN_FP32(exp)(float x) { return exp(x); }
__device__ inline float FN_FP32(erf)(float x) { return erf(x); }
__device__ inline float FN_FP32(sigmoid)(float x) { return 1.0f / (1.0f + exp(-x)); }
__device__ inline float FN_FP32(sqrt)(float x) { return sqrt(x); }
__device__ inline float FN_FP32(rsqrt)(float x) { return rsqrt(x); }
__device__ inline float FN_FP32(cbrt)(float x) { return cbrt(x); }

__device__ inline bool FN_FP32(isfinite)(float x) { return isfinite(x); }
__device__ inline bool FN_FP32(isinf)(float x) { return isinf(x); }
__device__ inline bool FN_FP32(isnan)(float x) { return isnan(x); }

__device__ inline float FN_FP32(pow)(float a, float b) { return powf(a, b); }

__device__ inline float FN_FP32(mod)(float a, float b) {
  float res = fmodf(a, b);
  if ((res != 0.0f) && ((res < 0.0f) != (b < 0.0f))) res += b;
  return res;
}

// *************************************************************** //
// float64 unary and binary operator
#define FN_FP64(func) cinn_nvgpu_##func##_fp64

__device__ inline double FN_FP64(sin)(double x) { return sin(x); }
__device__ inline double FN_FP64(cos)(double x) { return cos(x); }
__device__ inline double FN_FP64(tan)(double x) { return tan(x); }
__device__ inline double FN_FP64(sinh)(double x) { return sinh(x); }
__device__ inline double FN_FP64(cosh)(double x) { return cosh(x); }
__device__ inline double FN_FP64(tanh)(double x) { return tanh(x); }

__device__ inline double FN_FP64(asin)(double x) { return asin(x); }
__device__ inline double FN_FP64(acos)(double x) { return acos(x); }
__device__ inline double FN_FP64(atan)(double x) { return atan(x); }
__device__ inline double FN_FP64(asinh)(double x) { return asinh(x); }
__device__ inline double FN_FP64(acosh)(double x) { return acosh(x); }
__device__ inline double FN_FP64(atanh)(double x) { return atanh(x); }

__device__ inline double FN_FP64(ceil)(double x) { return ceil(x); }
__device__ inline double FN_FP64(round)(double x) { return round(x); }
__device__ inline double FN_FP64(trunc)(double x) { return trunc(x); }
__device__ inline double FN_FP64(abs)(double x) { return abs(x); }
__device__ inline double FN_FP64(floor)(double x) { return floor(x); }
__device__ inline double FN_FP64(log)(double x) { return log(x); }
__device__ inline double FN_FP64(log2)(double x) { return log2(x); }
__device__ inline double FN_FP64(log10)(double x) { return log10(x); }
__device__ inline double FN_FP64(exp)(double x) { return exp(x); }
__device__ inline double FN_FP64(erf)(double x) { return erf(x); }
__device__ inline double FN_FP64(sigmoid)(double x) { return 1.0 / (1.0 + exp(-x)); }
__device__ inline double FN_FP64(sqrt)(double x) { return sqrt(x); }
__device__ inline double FN_FP64(rsqrt)(double x) { return rsqrt(x); }
__device__ inline double FN_FP64(cbrt)(double x) { return cbrt(x); }

__device__ inline bool FN_FP64(isfinite)(double x) { return isfinite(x); }
__device__ inline bool FN_FP64(isinf)(double x) { return isinf(x); }
__device__ inline bool FN_FP64(isnan)(double x) { return isnan(x); }

__device__ inline double FN_FP64(pow)(double a, double b) { return pow(a, b); }
__device__ inline double FN_FP64(mod)(double a, double b) {
  double res = fmod(a, b);
  if ((res != 0.0) && ((res < 0.0) != (b < 0.0))) res += b;
  return res;
}

// *************************************************************** //
// int32 unary and binary operator
#define FN_INT32(func) cinn_nvgpu_##func##_int32

__device__ inline int FN_INT32(pow)(int a, int b) {
  int res = 1;
  for (int i = 0; i < b; ++i) {
    res *= a;
  }
  return res;
}

__device__ inline int FN_INT32(left_shift)(int a, int b) { return a << b; }
__device__ inline int FN_INT32(right_shift)(int a, int b) { return a >> b; }
__device__ inline int FN_INT32(bitwise_and)(int a, int b) { return a & b; }
__device__ inline int FN_INT32(bitwise_or)(int a, int b) { return a | b; }
__device__ inline int FN_INT32(bitwise_xor)(int a, int b) { return a ^ b; }
__device__ inline int FN_INT32(bitwise_not)(int a) { return ~a; }
__device__ inline int FN_INT32(clz)(int a) { return __clz(a); }
__device__ inline int FN_INT32(popc)(int a) { return __popc(a); }
__device__ inline int FN_INT32(logical_right_shift)(int a, int b) { return ((unsigned int)a >> b); }

__device__ inline int FN_INT32(max)(int a, int b) { return max(a, b); }
__device__ inline int FN_INT32(min)(int a, int b) { return min(a, b); }

__device__ inline int FN_INT32(mod)(int a, int b) {
  int res = a % b;
  if ((res != 0) && ((b ^ res) < 0)) res += b;
  return res;
}

// *************************************************************** //

// int64 unary and binary operator
#define FN_INT64(func) cinn_nvgpu_##func##_int64

__device__ inline long long int FN_INT64(bitwise_and)(long long int a, long long int b) { return a & b; }
__device__ inline long long int FN_INT64(bitwise_or)(long long int a, long long int b) { return a | b; }
__device__ inline long long int FN_INT64(bitwise_xor)(long long int a, long long int b) { return a ^ b; }
__device__ inline long long int FN_INT64(bitwise_not)(long long int a) { return ~a; }
__device__ inline long long int FN_INT64(clz)(long long int a) { return __clzll(a); }
__device__ inline long long int FN_INT64(popc)(long long int a) { return __popcll(a); }
__device__ inline long long int FN_INT64(mod)(long long int a, long long int b) {
  long long int res = a % b;
  if ((res != 0) && ((b ^ res) < 0)) res += b;
  return res;
}

__device__ inline long long int FN_INT64(pow)(long long int a, long long int b) {
  long long int res = 1;
  for (int i = 0; i < b; ++i) {
    res *= a;
  }
  return res;
}

// *************************************************************** //
// bfloat16 unary and binary operator
#ifdef CINN_CUDA_BF16

#define FN_BF16(func) cinn_nvgpu_##func##_bf16

__device__ inline bfloat16 FN_BF16(ceil)(bfloat16 x) { return bfloat16(hceil(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(floor)(bfloat16 x) { return bfloat16(hfloor(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(round)(bfloat16 x) { return bfloat16(hrint(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(trunc)(bfloat16 x) { return bfloat16(htrunc(x.to_nv_bfloat16())); }

__device__ inline bfloat16 FN_BF16(sin)(bfloat16 x) { return bfloat16(hsin(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(cos)(bfloat16 x) { return bfloat16(hcos(x.to_nv_bfloat16())); }

__device__ inline bfloat16 FN_BF16(exp)(bfloat16 x) { return bfloat16(hexp(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(log)(bfloat16 x) { return bfloat16(hlog(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(log2)(bfloat16 x) { return bfloat16(hlog2(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(log10)(bfloat16 x) { return bfloat16(hlog10(x.to_nv_bfloat16())); }

__device__ inline bfloat16 FN_BF16(sqrt)(bfloat16 x) { return bfloat16(hsqrt(x.to_nv_bfloat16())); }
__device__ inline bfloat16 FN_BF16(rsqrt)(bfloat16 x) { return bfloat16(hrsqrt(x.to_nv_bfloat16())); }

__device__ inline bfloat16 FN_BF16(cbrt)(bfloat16 x) { return bfloat16(FN_FP32(cbrt)(static_cast<float>(x))); }

__device__ inline bfloat16 FN_BF16(abs)(bfloat16 x) { return cinn::common::abs(x); }

__device__ inline bool FN_BF16(isnan)(bfloat16 x) { return cinn::common::isnan(x); }
__device__ inline bool FN_BF16(isinf)(bfloat16 x) { return cinn::common::isinf(x); }
__device__ inline bool FN_BF16(isfinite)(bfloat16 x) { return cinn::common::isfinite(x); }

__device__ inline bfloat16 FN_BF16(erf)(bfloat16 x) { return bfloat16(FN_FP32(erf)(static_cast<float>(x))); }

__device__ inline bfloat16 FN_BF16(tan)(bfloat16 x) { return bfloat16(FN_FP32(tan)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(sinh)(bfloat16 x) { return bfloat16(FN_FP32(sinh)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(cosh)(bfloat16 x) { return bfloat16(FN_FP32(cosh)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(tanh)(bfloat16 x) { return bfloat16(FN_FP32(tanh)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(asin)(bfloat16 x) { return bfloat16(FN_FP32(asin)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(acos)(bfloat16 x) { return bfloat16(FN_FP32(acos)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(atan)(bfloat16 x) { return bfloat16(FN_FP32(atan)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(asinh)(bfloat16 x) { return bfloat16(FN_FP32(asinh)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(acosh)(bfloat16 x) { return bfloat16(FN_FP32(acosh)(static_cast<float>(x))); }
__device__ inline bfloat16 FN_BF16(atanh)(bfloat16 x) { return bfloat16(FN_FP32(atanh)(static_cast<float>(x))); }

__device__ inline bfloat16 FN_BF16(sigmoid)(bfloat16 x) { return bfloat16(FN_FP32(sigmoid)(static_cast<float>(x))); }

__device__ inline bfloat16 FN_BF16(mod)(bfloat16 a, bfloat16 b) {
  return bfloat16(FN_FP32(mod)(static_cast<float>(a), static_cast<float>(b)));
}
__device__ inline bfloat16 FN_BF16(pow)(bfloat16 a, bfloat16 b) {
  return bfloat16(FN_FP32(pow)(static_cast<float>(a), static_cast<float>(b)));
}

#endif

// *************************************************************** //
// float16 unary and binary operator
#ifdef CINN_CUDA_FP16

#define FN_FP16(func) cinn_nvgpu_##func##_fp16

__device__ inline float16 FN_FP16(ceil)(float16 x) { return float16(hceil(x.to_half())); }
__device__ inline float16 FN_FP16(floor)(float16 x) { return float16(hfloor(x.to_half())); }
__device__ inline float16 FN_FP16(round)(float16 x) { return float16(hrint(x.to_half())); }
__device__ inline float16 FN_FP16(trunc)(float16 x) { return float16(htrunc(x.to_half())); }

__device__ inline float16 FN_FP16(sin)(float16 x) { return float16(hsin(x.to_half())); }
__device__ inline float16 FN_FP16(cos)(float16 x) { return float16(hcos(x.to_half())); }

__device__ inline float16 FN_FP16(exp)(float16 x) { return float16(hexp(x.to_half())); }
__device__ inline float16 FN_FP16(log)(float16 x) { return float16(hlog(x.to_half())); }
__device__ inline float16 FN_FP16(log2)(float16 x) { return float16(hlog2(x.to_half())); }
__device__ inline float16 FN_FP16(log10)(float16 x) { return float16(hlog10(x.to_half())); }

__device__ inline float16 FN_FP16(sqrt)(float16 x) { return float16(hsqrt(x.to_half())); }
__device__ inline float16 FN_FP16(rsqrt)(float16 x) { return float16(hrsqrt(x.to_half())); }

__device__ inline float16 FN_FP16(cbrt)(float16 x) { return float16(FN_FP32(cbrt)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(abs)(float16 x) { return cinn::common::abs(x); }

__device__ inline bool FN_FP16(isnan)(float16 x) { return cinn::common::isnan(x); }
__device__ inline bool FN_FP16(isinf)(float16 x) { return cinn::common::isinf(x); }
__device__ inline bool FN_FP16(isfinite)(float16 x) { return cinn::common::isfinite(x); }

__device__ inline float16 FN_FP16(erf)(float16 x) { return float16(FN_FP32(erf)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(tan)(float16 x) { return float16(FN_FP32(tan)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(sinh)(float16 x) { return float16(FN_FP32(sinh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(cosh)(float16 x) { return float16(FN_FP32(cosh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(tanh)(float16 x) { return float16(FN_FP32(tanh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(asin)(float16 x) { return float16(FN_FP32(asin)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(acos)(float16 x) { return float16(FN_FP32(acos)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(atan)(float16 x) { return float16(FN_FP32(atan)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(asinh)(float16 x) { return float16(FN_FP32(asinh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(acosh)(float16 x) { return float16(FN_FP32(acosh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(atanh)(float16 x) { return float16(FN_FP32(atanh)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(sigmoid)(float16 x) { return float16(FN_FP32(sigmoid)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(mod)(float16 a, float16 b) {
  return float16(FN_FP32(mod)(static_cast<float>(a), static_cast<float>(b)));
}
__device__ inline float16 FN_FP16(pow)(float16 a, float16 b) {
  return float16(FN_FP32(pow)(static_cast<float>(a), static_cast<float>(b)));
}

#endif

// *************************************************************** //
// reduce operator, need `--expt-relaxed-constexpr` option to call std function in device kernel
#define EXPAND_REDUCE_INT32_MARCO(MARCO, ...)       \
  MARCO(sum_int32, 0, int, ##__VA_ARGS__)           \
  MARCO(prod_int32, 1, int, ##__VA_ARGS__)          \
  MARCO(max_int32, CINN_INT32_MIN, int, ##__VA_ARGS__) \
  MARCO(min_int32, CINN_INT32_MAX, int, ##__VA_ARGS__)

__device__ inline int cinn_sum_int32(const int left, const int right) { return left + right; }
__device__ inline int cinn_prod_int32(const int left, const int right) { return left * right; }
__device__ inline int cinn_max_int32(const int left, const int right) { return max(left, right); }
__device__ inline int cinn_min_int32(const int left, const int right) { return min(left, right); }

#define EXPAND_REDUCE_INT64_MARCO(MARCO, ...)                          \
  MARCO(sum_int64, 0, long long int, ##__VA_ARGS__)                    \
  MARCO(prod_int64, 1, long long int, ##__VA_ARGS__)                   \
  MARCO(max_int64, -9223372036854775808, long long int, ##__VA_ARGS__) \
  MARCO(min_int64, 9223372036854775807, long long int, ##__VA_ARGS__)

__device__ inline long long int cinn_sum_int64(const long long int left, const long long int right) {
  return left + right;
}
__device__ inline long long int cinn_prod_int64(const long long int left, const long long int right) {
  return left * right;
}
__device__ inline long long int cinn_max_int64(const long long int left, const long long int right) {
  return max(left, right);
}
__device__ inline long long int cinn_min_int64(const long long int left, const long long int right) {
  return min(left, right);
}

#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...)           \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__)          \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__)         \
  MACRO(max_fp32, -3.40282e+38f, float, ##__VA_ARGS__) \
  MACRO(min_fp32, 3.40282e+38f, float, ##__VA_ARGS__)

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return max(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return min(left, right); }

#ifdef CINN_CUDA_BF16

#define EXPAND_REDUCE_BFP16_MACRO(MACRO, ...)                                           \
  MACRO(sum_bf16, bfloat16(0.0), bfloat16, ##__VA_ARGS__)                                \
  MACRO(prod_bf16, bfloat16(1.0), bfloat16, ##__VA_ARGS__)                               \
  MACRO(max_bf16, cinn::common::raw_uint16_to_bfloat16(0xfbff), bfloat16, ##__VA_ARGS__) \
  MACRO(min_bf16, cinn::common::raw_uint16_to_bfloat16(0x7bff), bfloat16, ##__VA_ARGS__)

__device__ inline bfloat16 cinn_sum_bf16(const bfloat16 left, const bfloat16 right) { return left + right; }
__device__ inline bfloat16 cinn_prod_bf16(const bfloat16 left, const bfloat16 right) { return left * right; }
__device__ inline bfloat16 cinn_max_bf16(const bfloat16 left, const bfloat16 right) { return max(left, right); }
__device__ inline bfloat16 cinn_min_bf16(const bfloat16 left, const bfloat16 right) { return min(left, right); }
#endif

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

#define EXPAND_REDUCE_FP64_MACRO(MACRO, ...)            \
  MACRO(sum_fp64, 0.0, double, ##__VA_ARGS__)           \
  MACRO(prod_fp64, 1.0, double, ##__VA_ARGS__)          \
  MACRO(max_fp64, -1.79769e+308, double, ##__VA_ARGS__) \
  MACRO(min_fp64, 1.79769e+308, double, ##__VA_ARGS__)

__device__ inline double cinn_sum_fp64(const double left, const double right) { return left + right; }
__device__ inline double cinn_prod_fp64(const double left, const double right) { return left * right; }
__device__ inline double cinn_max_fp64(const double left, const double right) { return max(left, right); }
__device__ inline double cinn_min_fp64(const double left, const double right) { return min(left, right); }

#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...) \
  MACRO(all, true, bool, ##__VA_ARGS__)      \
  MACRO(any, false, bool, ##__VA_ARGS__)

__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }

#define CINN_SHUFFLE_FUNCTION(offset, op, init)           \
  shfl_res = __shfl_down_sync(mask, tmp_val, offset, 32); \
  tmp_val  = op((threadIdx.x & 0x1f) + offset < lane ? shfl_res : init, tmp_val);

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                \
  __device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
    DTYPE tmp_val     = value, shfl_res;                                                  \
    unsigned int mask = __activemask();                                                   \
    unsigned int lane = __popc(mask);                                                     \
    if (lane < 32) {                                                                      \
      CINN_SHUFFLE_FUNCTION(16, cinn_##REDUCE_TYPE, (DTYPE)(INITIAL_VALUE))               \
      CINN_SHUFFLE_FUNCTION(8, cinn_##REDUCE_TYPE, (DTYPE)(INITIAL_VALUE))                \
      CINN_SHUFFLE_FUNCTION(4, cinn_##REDUCE_TYPE, (DTYPE)(INITIAL_VALUE))                \
      CINN_SHUFFLE_FUNCTION(2, cinn_##REDUCE_TYPE, (DTYPE)(INITIAL_VALUE))                \
      CINN_SHUFFLE_FUNCTION(1, cinn_##REDUCE_TYPE, (DTYPE)(INITIAL_VALUE))                \
      tmp_val = __shfl_sync(mask, tmp_val, 0, 32);                                        \
      return tmp_val;                                                                     \
    } else {                                                                              \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_down_sync(mask, tmp_val, 16, 32));     \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_down_sync(mask, tmp_val, 8, 32));      \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_down_sync(mask, tmp_val, 4, 32));      \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_down_sync(mask, tmp_val, 2, 32));      \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_down_sync(mask, tmp_val, 1, 32));      \
      return tmp_val;                                                                     \
    }                                                                                     \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_INT64_MARCO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP64_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
#endif

#undef CINN_WARP_SHUFFLE_INTERNAL_IMPL

#define CINN_WARP_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                     \
  __device__ inline DTYPE cinn_warp_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
    DTYPE tmp_val = (DTYPE)(INITIAL_VALUE);                                                          \
    for (int i = threadIdx.x; i < extend; i += 32) {                                                 \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]);                                        \
    }                                                                                                \
    return cinn_warp_shuffle_##REDUCE_TYPE##_internal(tmp_val);                                      \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_INT64_MARCO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_FP32_MACRO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_FP64_MACRO(CINN_WARP_REDUCE_IMPL)
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

#define CINN_BLOCK_REDUCE_INTERNAL_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                            \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE##_internal(const DTYPE value) {                              \
    CINN_BLOCK_REDUCE_INTERNAL_IMPL(DTYPE, value, (DTYPE)(INITIAL_VALUE), cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_INT64_MARCO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
#endif

#undef CINN_BLOCK_REDUCE_INTERNAL_IMPL
#undef CINN_BLOCK_REDUCE_INTERNAL_MACRO

#define CINN_BLOCK_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                                     \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
    DTYPE tmp_val = (DTYPE)(INITIAL_VALUE);                                                           \
    for (int i = threadIdx.x; i < extend; i += blockDim.x) {                                          \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]);                                         \
    }                                                                                                 \
    return cinn_block_reduce_##REDUCE_TYPE##_internal(tmp_val);                                       \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_INT64_MARCO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_FP64_MACRO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_IMPL)
#endif

#undef CINN_BLOCK_REDUCE_IMPL

#undef EXPAND_REDUCE_INT32_MARCO
#undef EXPAND_REDUCE_INT64_MARCO
#undef EXPAND_REDUCE_FP32_MACRO
#undef EXPAND_REDUCE_FP64_MACRO
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

__device__ inline int cinn_nvgpu_next_smallest_int32(int *buf, int size, int num, int begin, int stride) {
  int id = -1;
  for (int i = begin; i < begin + size * stride; i += stride) {
    if (id == -1 || buf[i] < buf[id]) {
      id = i;
    }
  }
  if (id != -1) {
    buf[id] = CINN_INT32_MAX;
    return (id - begin) / stride;
  }
  return -1;
}

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

#define CINN_NVGPU_LT_NUM(TYPE_SUFFIX, TYPE)                                                  \
  __device__ inline int cinn_nvgpu_lt_num_##TYPE_SUFFIX(                                      \
      const TYPE *buf, const int size, const TYPE num, const int offset, const int stride) { \
    int out = 0;                                                                             \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) {                   \
      if (buf[i] < num) out++;                                                               \
    }                                                                                        \
    return out;                                                                              \
  }

CINN_NVGPU_LT_NUM(fp32, float)
CINN_NVGPU_LT_NUM(fp64, double)
CINN_NVGPU_LT_NUM(int32, int)
CINN_NVGPU_LT_NUM(int64, long long int)

#undef CINN_NVGPU_LT_NUM

#define CINN_NVGPU_GT_NUM(TYPE_SUFFIX, TYPE)                                                  \
  __device__ inline int cinn_nvgpu_gt_num_##TYPE_SUFFIX(                                      \
      const TYPE *buf, const int size, const TYPE num, const int offset, const int stride) { \
    int out = 0;                                                                             \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) {                   \
      if (buf[i] > num) out++;                                                               \
    }                                                                                        \
    return out;                                                                              \
  }

CINN_NVGPU_GT_NUM(fp32, float)
CINN_NVGPU_GT_NUM(fp64, double)
CINN_NVGPU_GT_NUM(int32, int)
CINN_NVGPU_GT_NUM(int64, long long int)

#undef CINN_NVGPU_GT_NUM

#define CINN_NVGPU_INDEX_ADD(TYPE_SUFFIX, TYPE)                                \
  __device__ inline TYPE cinn_nvgpu_index_add_##TYPE_SUFFIX(const TYPE x,      \
                                            const int axis_indice,            \
                                            const TYPE *__restrict__ y,       \
                                            const int offset,                 \
                                            const int stride,                 \
                                            const int *__restrict__ index,    \
                                            const int index_size) {           \
    TYPE res = x;                                                             \
    int idx  = -1;                                                            \
    do {                                                                      \
      idx = cinn_cuda_find_int_from(index, index_size, axis_indice, idx + 1); \
      if (idx >= 0) {                                                         \
        res += y[offset + idx * stride];                                      \
      }                                                                       \
    } while (idx != -1);                                                      \
    return res;                                                               \
  }

CINN_NVGPU_INDEX_ADD(fp32, float)
CINN_NVGPU_INDEX_ADD(fp64, double)
#ifdef CINN_CUDA_FP16
CINN_NVGPU_INDEX_ADD(fp16, float16)
#endif

#undef CINN_CUDA_INDEX_ADD

__device__ int cinn_cuda_resize_bilinear(const int *buf,
                                         const int c_size,
                                         const int in_h,
                                         const int in_w,
                                         const int out_h,
                                         const int out_w,
                                         const int n,
                                         const int c,
                                         const int y,
                                         const int x) {
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y    = (y + 0.5F) * scale_y - 0.5F;
  float in_x    = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int  = static_cast<int>(FN_FP32(floor)(in_y));
  int in_x_int  = static_cast<int>(FN_FP32(floor)(in_x));
  float y_lerp  = in_y - in_y_int;
  float x_lerp  = in_x - in_x_int;
  float p[2][2];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int near_y = in_y_int + i;
      int near_x = in_x_int + j;
      near_y     = FN_INT32(max)(FN_INT32(min)(near_y, in_h - 1), 0);
      near_x     = FN_INT32(max)(FN_INT32(min)(near_x, in_w - 1), 0);
      p[i][j]    = buf[n * c_size * in_h * in_w + c * in_h * in_w + near_y * in_w + near_x];
    }
  }

  float top    = p[0][0] * (1.0F - x_lerp) + p[0][1] * x_lerp;
  float bottom = p[1][0] * (1.0F - x_lerp) + p[1][1] * x_lerp;
  float value  = top * (1.0F - y_lerp) + bottom * y_lerp;
  return value;
}

__device__ int cinn_cuda_resize_bicubic(const int *buf,
                                        const int c_size,
                                        const int in_h,
                                        const int in_w,
                                        const int out_h,
                                        const int out_w,
                                        const int n,
                                        const int c,
                                        const int y,
                                        const int x) {
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y    = (y + 0.5F) * scale_y - 0.5F;
  float in_x    = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int  = static_cast<int>(cinn_nvgpu_floor_fp32(in_y));
  int in_x_int  = static_cast<int>(cinn_nvgpu_floor_fp32(in_x));
  float y_fract = in_y - cinn_nvgpu_floor_fp32(in_y);
  float x_fract = in_x - cinn_nvgpu_floor_fp32(in_x);
  float p[4][4];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int near_y = in_y_int + i - 1;
      int near_x = in_x_int + j - 1;
      near_y     = FN_INT32(max)(FN_INT32(min)(near_y, in_h - 1), 0);
      near_x     = FN_INT32(max)(FN_INT32(min)(near_x, in_w - 1), 0);
      p[i][j]    = buf[n * c_size * in_h * in_w + c * in_h * in_w + near_y * in_w + near_x];
    }
  }

  float alpha = -0.5F;
  float w[2][4];

  for (int i = 0; i < 2; ++i) {
    float t  = (i == 0 ? x_fract : y_fract);
    float t2 = t * t;
    float t3 = t * t * t;
    w[i][0]  = alpha * (t3 - 2 * t2 + t);
    w[i][1]  = (alpha + 2) * t3 - (3 + alpha) * t2 + 1;
    w[i][2]  = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t;
    w[i][3]  = -alpha * t3 + alpha * t2;
  }

  float col[4];

  for (int i = 0; i < 4; ++i) {
    col[i] = 0.0F;
    for (int j = 0; j < 4; ++j) {
      col[i] += p[i][j] * w[0][j];
    }
  }

  float value = 0.0F;

  for (int i = 0; i < 4; ++i) {
    value += col[i] * w[1][i];
  }

  return value;
}

// *************************************************************** //
// end of macro undef
#undef CINN_INT32_MAX
#undef CINN_INT32_MIN
#undef FN_BOOL
#undef FN_UINT8
#undef FN_INT8
#undef FN_INT16
#undef FN_FP32
#undef FN_FP64
#undef FN_INT32
#undef FN_INT64

#ifdef CINN_CUDA_FP16
#undef FN_FP16
#endif
}

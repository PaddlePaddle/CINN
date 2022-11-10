#ifdef CINN_CUDA_FP16

#define FN_FP16(func) cinn_nvgpu_##func##_fp16

__device__ inline float16 FN_FP16(ceil)(float16 x) { return hceil(x.to_half()); }
__device__ inline float16 FN_FP16(floor)(float16 x) { return hfloor(x.to_half()); }
__device__ inline float16 FN_FP16(round)(float16 x) { return hrint(x.to_half()); }
__device__ inline float16 FN_FP16(trunc)(float16 x) { return htrunc(x.to_half()); }

__device__ inline float16 FN_FP16(sin)(float16 x) { return hsin(x.to_half()); }
__device__ inline float16 FN_FP16(cos)(float16 x) { return hcos(x.to_half()); }

__device__ inline float16 FN_FP16(exp)(float16 x) { return hexp(x.to_half()); }
__device__ inline float16 FN_FP16(log)(float16 x) { return hlog(x.to_half()); }
__device__ inline float16 FN_FP16(log2)(float16 x) { return hlog2(x.to_half()); }
__device__ inline float16 FN_FP16(log10)(float16 x) { return hlog10(x.to_half()); }

__device__ inline float16 FN_FP16(sqrt)(float16 x) { return hsqrt(x.to_half()); }
__device__ inline float16 FN_FP16(rsqrt)(float16 x) { return hrsqrt(x.to_half()); }

__device__ inline float16 FN_FP16(abs)(float16 x) { return abs(x); }

__device__ inline bool FN_FP16(isnan)(float16 x) { return isnan(x); }
__device__ inline bool FN_FP16(isinf)(float16 x) { return isinf(x); }
__device__ inline bool FN_FP16(isfinite)(float16 x) { return isfinite(x); }

__device__ inline float16 FN_FP16(max)(float16 a, float16 b) { return __hmax_nan(x.to_half()); }
__device__ inline float16 FN_FP16(min)(float16 a, float16 b) { return __hmin_nan(x.to_half()); }

#define FN_FP32(func) cinn_nvgpu_##func##_fp32
__device__ inline float16 FN_FP16(erf)(float16 x) { return FN_FP32(erf)(static_cast<float>(x)); }

__device__ inline float16 FN_FP16(pow)(float16 x) { return FN_FP32(pow)(static_cast<float>(x)); }

__device__ inline float16 FN_FP16(cosh)(float16 x) { return FN_FP32(cosh)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(tanh)(float16 x) { return FN_FP32(tanh)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(asin)(float16 x) { return FN_FP32(asin)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(acos)(float16 x) { return FN_FP32(acos)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(acosh)(float16 x) { return FN_FP32(acosh)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(atanh)(float16 x) { return FN_FP32(atanh)(static_cast<float>(x)); }

__device__ inline float16 FN_FP16(sigmoid)(float16 x) { return FN_FP32(sigmoid)(static_cast<float>(x)); }
__device__ inline float16 FN_FP16(remainder)(float16 x) { return FN_FP32(remainder)(static_cast<float>(x)); }
#undef FN_FP32

#undef FN_FP16

#endif

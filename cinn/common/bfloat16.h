// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#ifndef CINN_COMMON_BFLOAT16_H
#define CINN_COMMON_BFLOAT16_H

#ifdef __cplusplus
#pragma once
#endif  // __cplusplus

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#ifdef CINN_WITH_CUDA
#include <cuda.h>

#if (defined(__CUDACC__) || defined(__CUDACC_RTC__)) && CUDA_VERSION >= 11000
#define CINN_CUDA_BF16
#include <cuda_bf16.h>

#endif  // __CUDACC__
#endif  // CINN_WITH_CUDA

#ifdef __cplusplus

#ifndef _WIN32
#define CINN_ALIGN(x) __attribute__((aligned(x)))
#else  // _WIN32
#define CINN_ALIGN(x) __declspec(align(x))
#endif  // _WIN32

#else  // __cplusplus
#define CINN_ALIGN(x)
#endif  // __cplusplus

// The `HOST` macro definition is not used here, it has a potential
// conflict with the enumeration `kHOST` representing the backend.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
namespace cinn {
namespace common {
#endif  // __cplusplus

// Use CINN_ALIGNED(2) to ensure that each bfloat16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes bfloat16 compatible
// with CUDA half
struct CINN_ALIGN(2) bfloat16 {
  uint16_t x;

#ifdef __cplusplus
  // Constructors
  bfloat16()                  = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o)                 = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16()                       = default;

  __host__ __device__ inline explicit bfloat16(float val) {
#if defined(CINN_CUDA_BF16)
    __nv_bfloat16 tmp = __float2bfloat16(val);
    x                 = *reinterpret_cast<uint16_t*>(&tmp);
#else
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
#endif
  }

#if defined(CINN_CUDA_BF16)
  __host__ __device__ inline explicit bfloat16(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
  }
#endif

  template <class T>
  __host__ __device__ inline explicit bfloat16(const T& val) : x(bfloat16(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(CINN_CUDA_BF16)
  __host__ __device__ inline bfloat16& operator=(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
    return *this;
  }
#endif

  __host__ __device__ inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  __host__ __device__ inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  // Conversion opertors
  __host__ __device__ inline operator float() const {
#ifdef CINN_CUDA_BF16
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
    float val     = 0.f;
    uint16_t temp = x;
    std::memcpy(reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp), 2);
    return val;
#endif
  }

#ifdef CINN_CUDA_BF16
  __host__ __device__ inline __nv_bfloat16 to_nv_bfloat16() const {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif

  __host__ __device__ inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  __host__ __device__ inline explicit operator int8_t() const { return static_cast<int8_t>(static_cast<float>(*this)); }

  __host__ __device__ inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline operator double() const { return static_cast<double>(static_cast<float>(*this)); }
#endif  // __cplusplus
};

__host__ __device__ inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

__host__ __device__ inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

__host__ __device__ inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

__host__ __device__ inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

__host__ __device__ inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

__host__ __device__ inline bfloat16& operator+=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

__host__ __device__ inline bfloat16& operator-=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

__host__ __device__ inline bfloat16& operator*=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

__host__ __device__ inline bfloat16& operator/=(bfloat16& a,  // NOLINT
                                                const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

__host__ __device__ inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

// Comparison operators
__host__ __device__ inline bool operator==(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

__host__ __device__ inline bool operator!=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

__host__ __device__ inline bool operator<(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

__host__ __device__ inline bool operator<=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

__host__ __device__ inline bool operator>(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

__host__ __device__ inline bool operator>=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

__host__ __device__ inline bool(isnan)(const bfloat16& a) { return (a.x & 0x7FFF) > 0x7F80; }

__host__ __device__ inline bool(isinf)(const bfloat16& a) { return (a.x & 0x7F80) == 0x7F80; }

__host__ __device__ inline bool(isfinite)(const bfloat16& a) { return !((isnan)(a)) && !((isinf)(a)); }

__host__ __device__ inline bfloat16(abs)(const bfloat16& a) { return bfloat16(std::abs(static_cast<float>(a))); }

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << static_cast<float>(a);
  return os;
}

#ifdef __cplusplus
}  // namespace common
}  // namespace cinn
#endif  // __cplusplus

namespace std {

// Override some std func (e.g. std::is_pod::value) for bfloat16
template <>
struct is_pod<cinn::common::bfloat16> {
  static const bool value =
      is_trivial<cinn::common::bfloat16>::value && is_standard_layout<cinn::common::bfloat16>::value;
};

template <>
struct is_floating_point<cinn::common::bfloat16>
    : std::integral_constant<
          bool,
          std::is_same<cinn::common::bfloat16, typename std::remove_cv<cinn::common::bfloat16>::type>::value> {};
template <>
struct is_signed<cinn::common::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<cinn::common::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const cinn::common::bfloat16& a) { return cinn::common::isnan(a); }

inline bool isinf(const cinn::common::bfloat16& a) { return cinn::common::isinf(a); }

template <>
struct numeric_limits<cinn::common::bfloat16> {
  static const bool is_specialized                = true;
  static const bool is_signed                     = true;
  static const bool is_integer                    = false;
  static const bool is_exact                      = false;
  static const bool has_infinity                  = true;
  static const bool has_quiet_NaN                 = true;
  static const bool has_signaling_NaN             = true;
  static const float_denorm_style has_denorm      = denorm_present;
  static const bool has_denorm_loss               = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559                     = false;
  static const bool is_bounded                    = false;
  static const bool is_modulo                     = false;
  static const int digits                         = 8;
  static const int digits10                       = 2;
  static const int max_digits10                   = 9;
  static const int radix                          = 2;
  static const int min_exponent                   = -125;
  static const int min_exponent10                 = -37;
  static const int max_exponent                   = 128;
  static const int max_exponent10                 = 38;
  static const bool traps                         = true;
  static const bool tinyness_before               = false;

  __host__ __device__ static cinn::common::bfloat16(min)() { return cinn::common::raw_uint16_to_bfloat16(0x007f); }
  __host__ __device__ static cinn::common::bfloat16 lowest() { return cinn::common::raw_uint16_to_bfloat16(0xff7f); }
  __host__ __device__ static cinn::common::bfloat16(max)() { return cinn::common::raw_uint16_to_bfloat16(0x7f7f); }
  __host__ __device__ static cinn::common::bfloat16 epsilon() { return cinn::common::raw_uint16_to_bfloat16(0x3400); }
  __host__ __device__ static cinn::common::bfloat16 round_error() { return cinn::common::bfloat16(0.5); }
  __host__ __device__ static cinn::common::bfloat16 infinity() { return cinn::common::raw_uint16_to_bfloat16(0x7f80); }
  __host__ __device__ static cinn::common::bfloat16 quiet_NaN() { return cinn::common::raw_uint16_to_bfloat16(0xffc1); }
  __host__ __device__ static cinn::common::bfloat16 signaling_NaN() {
    return cinn::common::raw_uint16_to_bfloat16(0xff81);
  }
  __host__ __device__ static cinn::common::bfloat16 denorm_min() {
    return cinn::common::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std

#endif  // CINN_COMMON_BFLOAT16_H

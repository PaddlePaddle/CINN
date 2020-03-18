#include <immintrin.h>
#include <stdint.h>

#include <vector>

//! Vector in stack, this can only used in generated .cc file.
template <typename T, size_t Num>
struct StackVec {
  typedef T value_type;
  typedef StackVec<T, Num> self_type;

  self_type& operator=(const StackVec& src) {
    if (this != &src) {
      memcpy(data_, src.data_, num_bytes());
    }
    return *this;
  }

  StackVec() { memset(data_, 0, num_bytes()); }

  StackVec(const T* externl) : external_data_(externl) {}

  static self_type Broadcast(const value_type& v) {
    self_type res;
    for (size_t i = 0; i < Num; i++) res.data_[i] = v;
    return res;
  }

  static self_type Ramp(const value_type& base, const value_type& stride) {
    self_type res;
    for (size_t i = 0; i < Num; i++) {
      res.data_[i] = base + stride * i;
    }
  }

  static self_type Load(const void* base, int32_t offset) {
    self_type res;
    memcpy(&res.data_[0], (const value_type*)base + offset, num_bytes());
  }

  static self_type Load(const void* base, const StackVec<int32_t, Num>& offset) {
    self_type res;
    for (size_t i = 0; i < Num; i++) {
      res.data_[i] = ((const value_type*)base)[offset[i]];
    }
  }

  void Store(void* base, int32_t offset) const { mempcpy((value_type*)base + offset, &data_[0], num_bytes()); }

  inline value_type& operator[](size_t i) { return data_[i]; }
  inline value_type operator[](size_t i) const { return data_[i]; }

  // binary operator between two vectors
  // @{
#define __(op__)                                                           \
  friend self_type operator op__(const self_type& a, const self_type& b) { \
    self_type res;                                                         \
    for (size_t i = 0; i < Num; i++) {                                     \
      res.data_[i] = a[i] op__ b[i];                                       \
    }                                                                      \
    return res;                                                            \
  }
  __(+)
  __(-)
  __(*)
  __(/)
  __(%)
  // @}
#undef __

  // binary operator between a vector and a scalar
  // @{
#define __(op__)                                                            \
  friend self_type operator op__(const self_type& a, const value_type& b) { \
    self_type res;                                                          \
    for (size_t i = 0; i < Num; i++) {                                      \
      res.data_[i] = a[i] op__ b;                                           \
    }                                                                       \
    return res;                                                             \
  }
  __(+)
  __(-)
  __(*)
  __(/)
  __(%)
#undef __
  // @}

  static constexpr size_t num_bytes() { return sizeof(data_); }

 private:
  T data_[Num];
  T* external_data_{nullptr};
};

/**
 * The vector with external data.
 */
template <typename T, size_t Num>
struct ExternalVec {
  typedef T value_type;
  typedef ExternalVec<T, Num> self_type;

  ExternalVec(T* data) : data_(data) {}

  self_type& operator=(const self_type& src) {
    if (data_ != src.data_) {
      memcpy(data_, src.data_, num_bytes());
    }
    return *this;
  }

  static self_type Load(const void* base, int32_t offset) {
    self_type res((T*)base + offset);
    return res;
  }

  static constexpr size_t num_bytes() { return sizeof(value_type) * Num; }

 private:
  T* data_{nullptr};
};

// AVX256 load
//@{
inline __m256 cinn_avx256_load(float* dst) { return _mm256_load_ps(dst); }
inline __m256 cinn_avx256_load(double* dst) { return _mm256_load_pd(dst); }
//@}
// AVX512 load
//@{
inline __m512 cinn_avx512_load(float* dst) { return _mm512_load_ps(dst); }
inline __m512 cinn_avx512_load(double* dst) { return _mm512_load_pd(dst); }
//@}

// FP32x8 * FP32x8
// @{
inline void cinn_avx256_add(float* dst, float* a, float* b) {
  _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_sub(float* dst, float* a, float* b) {
  _mm256_store_ps(dst, _mm256_sub_ps(_mm256_load_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_mul(float* dst, float* a, float* b) {
  _mm256_store_ps(dst, _mm256_mul_ps(_mm256_load_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_div(float* dst, float* a, float* b) {
  _mm256_store_ps(dst, _mm256_div_ps(_mm256_load_ps(a), _mm256_load_ps(b)));
}
// @}

// FP32x4 * float
// @{
inline void cinn_avx256_add(float* dst, float* a, float b) {
  _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(a), _mm256_set1_ps(b)));
}
inline void cinn_avx256_sub(float* dst, float* a, float b) {
  _mm256_store_ps(dst, _mm256_sub_ps(_mm256_load_ps(a), _mm256_set1_ps(b)));
}
inline void cinn_avx256_mul(float* dst, float* a, float b) {
  _mm256_store_ps(dst, _mm256_mul_ps(_mm256_load_ps(a), _mm256_set1_ps(b)));
}
inline void cinn_avx256_div(float* dst, float* a, float b) {
  _mm256_store_ps(dst, _mm256_div_ps(_mm256_load_ps(a), _mm256_set1_ps(b)));
}
// @}

// float * FP32x4
// @{
inline void cinn_avx256_add(float* dst, float a, float* b) {
  _mm256_store_ps(dst, _mm256_add_ps(_mm256_set1_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_sub(float* dst, float a, float* b) {
  _mm256_store_ps(dst, _mm256_sub_ps(_mm256_set1_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_mul(float* dst, float a, float* b) {
  _mm256_store_ps(dst, _mm256_mul_ps(_mm256_set1_ps(a), _mm256_load_ps(b)));
}
inline void cinn_avx256_div(float* dst, float a, float* b) {
  _mm256_store_ps(dst, _mm256_div_ps(_mm256_set1_ps(a), _mm256_load_ps(b)));
}
// @}

// 4 x float64
// @{
inline void cinn_avx256_add(double* dst, double* a, double* b) {
  _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_sub(double* dst, double* a, double* b) {
  _mm256_store_pd(dst, _mm256_sub_pd(_mm256_load_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_mul(double* dst, double* a, double* b) {
  _mm256_store_pd(dst, _mm256_mul_pd(_mm256_load_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_div(double* dst, double* a, double* b) {
  _mm256_store_pd(dst, _mm256_div_pd(_mm256_load_pd(a), _mm256_load_pd(b)));
}
// @}

// FP32x4 * FP64
// @{
inline void cinn_avx256_add(double* dst, double* a, double b) {
  _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(a), _mm256_set1_pd(b)));
}
inline void cinn_avx256_sub(double* dst, double* a, double b) {
  _mm256_store_pd(dst, _mm256_sub_pd(_mm256_load_pd(a), _mm256_set1_pd(b)));
}
inline void cinn_avx256_mul(double* dst, double* a, double b) {
  _mm256_store_pd(dst, _mm256_mul_pd(_mm256_load_pd(a), _mm256_set1_pd(b)));
}
inline void cinn_avx256_div(double* dst, double* a, double b) {
  _mm256_store_pd(dst, _mm256_div_pd(_mm256_load_pd(a), _mm256_set1_pd(b)));
}
// @}

// float * FP32x4
// @{
inline void cinn_avx256_add(double* dst, double a, double* b) {
  _mm256_store_pd(dst, _mm256_add_pd(_mm256_set1_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_sub(double* dst, double a, double* b) {
  _mm256_store_pd(dst, _mm256_sub_pd(_mm256_set1_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_mul(double* dst, double a, double* b) {
  _mm256_store_pd(dst, _mm256_mul_pd(_mm256_set1_pd(a), _mm256_load_pd(b)));
}
inline void cinn_avx256_div(double* dst, double a, double* b) {
  _mm256_store_pd(dst, _mm256_div_pd(_mm256_set1_pd(a), _mm256_load_pd(b)));
}
// @}

//! 32 x float32 operations.
// @{
inline void cinn_avx512_add(float* dst, float* a, float* b) {
  _mm512_store_ps(dst, _mm512_add_ps(_mm512_load_ps(a), _mm512_load_ps(b)));
}
inline void cinn_avx512_sub(float* dst, float* a, float* b) {
  _mm512_store_ps(dst, _mm512_sub_ps(_mm512_load_ps(a), _mm512_load_ps(b)));
}
inline void cinn_avx512_mul(float* dst, float* a, float* b) {
  _mm512_store_ps(dst, _mm512_mul_ps(_mm512_load_ps(a), _mm512_load_ps(b)));
}
inline void cinn_avx512_div(float* dst, float* a, float* b) {
  _mm512_store_ps(dst, _mm512_div_ps(_mm512_load_ps(a), _mm512_load_ps(b)));
}
// @}

// FP32x4 * FP64
// @{
inline void cinn_avx512_add(float* dst, float* a, float b) {
  _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(a), _mm512_set1_pd(b)));
}
inline void cinn_avx512_sub(float* dst, float* a, float b) {
  _mm512_store_pd(dst, _mm512_sub_pd(_mm512_load_pd(a), _mm512_set1_pd(b)));
}
inline void cinn_avx512_mul(float* dst, float* a, float b) {
  _mm512_store_pd(dst, _mm512_mul_pd(_mm512_load_pd(a), _mm512_set1_pd(b)));
}
inline void cinn_avx512_div(float* dst, float* a, float b) {
  _mm512_store_pd(dst, _mm512_div_pd(_mm512_load_pd(a), _mm512_set1_pd(b)));
}
// @}

// float * FP32x4
// @{
inline void cinn_avx512_add(float* dst, float a, float* b) {
  _mm512_store_pd(dst, _mm512_add_pd(_mm512_set1_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_sub(float* dst, float a, float* b) {
  _mm512_store_pd(dst, _mm512_sub_pd(_mm512_set1_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_mul(float* dst, float a, float* b) {
  _mm512_store_pd(dst, _mm512_mul_pd(_mm512_set1_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_div(float* dst, float a, float* b) {
  _mm512_store_pd(dst, _mm512_div_pd(_mm512_set1_pd(a), _mm512_load_pd(b)));
}
// @}

//! 16 x float32 operations.
// @{
inline void cinn_avx512_add(double* dst, double* a, double* b) {
  _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_sub(double* dst, double* a, double* b) {
  _mm512_store_pd(dst, _mm512_sub_pd(_mm512_load_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_mul(double* dst, double* a, double* b) {
  _mm512_store_pd(dst, _mm512_mul_pd(_mm512_load_pd(a), _mm512_load_pd(b)));
}
inline void cinn_avx512_div(double* dst, double* a, double* b) {
  _mm512_store_pd(dst, _mm512_div_pd(_mm512_load_pd(a), _mm512_load_pd(b)));
}
// @}

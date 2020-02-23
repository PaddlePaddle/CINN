#ifndef CINN_RUNTIME_H_
#define CINN_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CINN_ALWAYS_INLINE __attribute__((always_inline)) inline

typedef enum cinn_type_code_t {
  cinn_type_int    = 0,  //! signed int
  cinn_type_uint   = 1,  //! unsigned int
  cinn_type_float  = 1,  //! floating point
  cinn_type_handle = 1   //! void*
} cinn_type_code_t;

#ifndef CINN_ATTRIBUTE_ALIGN
#define CINN_ATTRIBUTE_ALIGN(n) __attribute__((aligned(n)))
#endif

/**
 * A tuntime tag for type in CINN system.
 */
typedef struct cinn_type_t {
#if __cplusplus >= 201103L
  CINN_ATTRIBUTE_ALIGN(1) cinn_type_code_t code;
#else
  uint8_t code;
#endif

  //! Number of bits.
  uint8_t bits;

  //! Number of elements in a vector, 1 for scalar.
  uint16_t lanes;

#ifdef __cplusplus
  CINN_ALWAYS_INLINE cinn_type_t() : code(cinn_type_int), bits(0), lanes(0) {}
  CINN_ALWAYS_INLINE cinn_type_t(cinn_type_code_t code, uint8_t bits, uint16_t lanes = 1)
      : code(code), bits(bits), lanes(lanes) {}
  CINN_ALWAYS_INLINE bool operator==(const cinn_type_t& other) const {
    return code == other.code && bits == other.bits && lanes == other.lanes;
  }
  CINN_ALWAYS_INLINE bool operator!=(const cinn_type_t& other) const { return !(*this == other); }
  CINN_ALWAYS_INLINE uint16_t bytes() const { return (bits + 7) / 8; }
#endif  // __cplusplus

} cinn_type_t;

//! Help to define the size of a dimension, due to polyhedral representation, we no need to record the extend or
//! min(default to 0).
typedef int cinn_dimension_t;

//! Help to tell where the buffer locates.
typedef enum cinn_buffer_kind_t {
  cinn_buffer_on_host   = 0,      //! buffer on host
  cinn_buffer_on_device = 1 << 1  // ! buffer on device e.g. GPU.
} cinn_buffer_kind_t;

//! The raw representation of a buffer,used in the generated code/lib.
typedef struct cinn_buffer_t {
  //! A device handle.
  uint64_t device;

  //! A pointer to the memory in host.
  uint8_t* host_memory;

  //! Extra flags.
  uint64_t flag;

  //! Data type.
  cinn_type_t type;

  //! Number of dimensions.
  int32_t ndims;
  cinn_buffer_t* dims;

#ifdef __cplusplus
  int num_elements() const {
    int res = 1;
    for (int i = 0; i < ndims; i++) {
      res *= dims[i];
    }
    return res;
  }

  CINN_ALWAYS_INLINE bool on_host() const { return get_flag(cinn_buffer_on_host); }
  CINN_ALWAYS_INLINE bool on_device() const { return get_flag(cinn_buffer_on_device); }
  CINN_ALWAYS_INLINE void set_on_host(bool x = true) {
    if (x) {
      set_flag(cinn_buffer_on_host);
    } else {
      flag &= ~cinn_buffer_on_host;
    }
  }
  CINN_ALWAYS_INLINE void set_on_device(bool x = true) {
    if (x) {
      set_flag(cinn_buffer_on_device);
    } else {
      flag &= ~cinn_buffer_on_device;
    }
  }
  CINN_ALWAYS_INLINE uint8_t* begin() const {}

  CINN_ALWAYS_INLINE bool get_flag(cinn_buffer_kind_t flag) const { return (this->flag & flag) != 0; }
  CINN_ALWAYS_INLINE void set_flag(cinn_buffer_kind_t flag) { this->flag |= flag; }

#endif  // __cplusplus

} cinn_buffer_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CINN_RUNTIME_H_

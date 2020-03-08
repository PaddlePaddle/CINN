#ifndef CINN_RUNTIME_CINN_RUNTIME_H_
#define CINN_RUNTIME_CINN_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CINN_ALWAYS_INLINE __attribute__((always_inline)) inline

//! Code for the primitive types supported in CINN.
typedef enum cinn_type_code_t {
  cinn_type_unk    = -1,  //! Unknown type
  cinn_type_int    = 0,   //! signed int
  cinn_type_uint   = 1,   //! unsigned int
  cinn_type_float  = 2,   //! floating point
  cinn_type_handle = 3    //! void*
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

//! Some primitive types.
// @{
extern cinn_type_t cinn_unk_t();
extern cinn_type_t cinn_int32_t();
extern cinn_type_t cinn_int64_t();
extern cinn_type_t cinn_uint32_t();
extern cinn_type_t cinn_uint64_t();
extern cinn_type_t cinn_float32_t();
extern cinn_type_t cinn_float64_t();
// @}

//! Help to define the size of a dimension, due to polyhedral representation, we no need to record the extend or
//! min(default to 0).
typedef int cinn_dimension_t;

//! Help to tell the kind of the device.
typedef enum cinn_device_kind_t {
  cinn_unk_device    = -1,  // Undefined device.
  cinn_x86_device    = 0,   // X86 device
  cinn_opencl_device = 1,   // OpenCL device
  cinn_arm_device    = 2    // ARM device
} cinn_device_kind_t;

//! Help to tell where the buffer locates.
typedef enum cinn_buffer_kind_t {
  cinn_buffer_on_host   = 0,      //! buffer on host
  cinn_buffer_on_device = 1 << 1  // ! buffer on device e.g. GPU.
} cinn_buffer_kind_t;

struct cinn_buffer_t;

/**
 * All CINN backends implementation should provide an interface to be used.
 */
struct cinn_device_interface_impl_t;

struct cinn_device_interface_t {
  int (*malloc)(void* context, struct cinn_buffer_t* buf);
  int (*free)(void* context, struct cinn_buffer_t* buf);
  int (*sync)(void* context, struct cinn_buffer_t* buf);
  int (*release)(void* context, const struct cinn_device_interface_t* device_interface);
  int (*copy_to_host)(void* context, struct cinn_buffer_t* buf);
  int (*copy_to_device)(void* context, struct cinn_buffer_t* buf);
  int (*buffer_copy)(void* context, struct cinn_buffer_t* src, struct cinn_buffer_t* dst);
  cinn_device_interface_impl_t* impl;
};

/**
 * Release all data associated with the given interface.
 */
extern int cinn_device_release(void* context, const struct cinn_device_interface_t* device_interface);

/*
 * Copy image data from device to host memory.
 */
extern int cinn_buffer_copy_to_host(void* context, struct cinn_buffer_t* buf);

//! Copy data from host to device memory.
extern int cinn_buffer_copy_to_device(void* context, struct cinn_buffer_t* buf);

//! Copy data from one buffer to another.
extern int cinn_buffer_copy(void* context, struct cinn_buffer_t* src, struct cinn_buffer_t* dst);

//! Wait for current device operations to complete.
extern int cinn_device_sync(void* context, struct cinn_buffer_t* buf);

//! Allocate device memory.
extern int cinn_buffer_malloc(void* context, struct cinn_buffer_t* buf);

//! Free device memory.
extern int cinn_buffer_free(void* context, struct cinn_buffer_t* buf);

//! Get the memory address in buffer.
extern void* cinn_buffer_get_data_handle(struct cinn_buffer_t* buf);
extern void* cinn_buffer_get_data_const_handle(const struct cinn_buffer_t* buf);

//! The raw representation of a buffer,used in the generated code/lib.
typedef struct cinn_buffer_t {
  //! Tell which kind of device this buffer locates.
  cinn_device_kind_t device;

  //! The interface used to operate on device.
  const struct cinn_device_interface_t* device_interface;

  //! A pointer to the memory in host.
  uint8_t* host_memory;

  //! Extra flags.
  uint64_t flag;

  //! Data type.
  cinn_type_t type;

  //! Number of dimensions.
  int32_t dimensions;
  cinn_dimension_t* dims;

  //! The actual memory size.
  uint64_t memory_size;

#ifdef __cplusplus
  cinn_buffer_t()
      : device(cinn_unk_device),
        device_interface(NULL),
        host_memory(NULL),
        flag(0UL),
        type(cinn_type_t()),
        dimensions(0),
        dims(NULL),
        memory_size(0) {}

  static struct cinn_buffer_t* new_(cinn_device_kind_t device, cinn_type_t type);
  static void delete_(struct cinn_buffer_t* x) { delete x; }

  // NOTE the buffer should be resized first.
  static void alloc(struct cinn_buffer_t*);

  //! Set the shape of the buffer. NOTE this just record the shape, not allocate the memory.
  CINN_ALWAYS_INLINE void resize(const cinn_dimension_t* dims, int dimensions) {
    if (this->dimensions != dimensions) {
      if (this->dims) free(this->dims);
      this->dims = (cinn_dimension_t*)malloc(dimensions * sizeof(cinn_dimension_t));
    }
    this->dimensions = dimensions;
    memcpy(this->dims, dims, dimensions * sizeof(cinn_dimension_t));
  }

  CINN_ALWAYS_INLINE int num_elements() const {
    int res = 1;
    for (int i = 0; i < dimensions; i++) {
      res *= dims[i];
    }
    return res;
  }

  CINN_ALWAYS_INLINE bool on_host() const { return get_flag(cinn_buffer_on_host); }
  CINN_ALWAYS_INLINE bool on_device() const { return get_flag(cinn_buffer_on_device); }
  CINN_ALWAYS_INLINE void set_on_host(bool x = true) { set_flag(cinn_buffer_on_host, x); }
  CINN_ALWAYS_INLINE void set_on_device(bool x = true) { set_flag(cinn_buffer_on_device, x); }

  CINN_ALWAYS_INLINE int device_sync(void* ctx = NULL) {
    if (device_interface && device_interface->sync) {
      return device_interface->sync(ctx, this);
    }
    return 0;
  }

  CINN_ALWAYS_INLINE uint8_t* begin() const { return 0; }
  CINN_ALWAYS_INLINE uint8_t* end() const { return host_memory + num_elements() * type.bytes(); }

  CINN_ALWAYS_INLINE bool get_flag(cinn_buffer_kind_t flag) const { return (this->flag & flag) != 0; }
  CINN_ALWAYS_INLINE void set_flag(cinn_buffer_kind_t flag, bool value) {
    if (value)
      this->flag |= flag;
    else
      this->flag &= ~flag;
  }

#endif  // __cplusplus

} cinn_buffer_t;

struct cinn_device_interface_impl_t {
  int (*malloc)(void* context, struct cinn_buffer_t* buf);
  int (*free)(void* context, struct cinn_buffer_t* buf);
  int (*sync)(void* context, struct cinn_buffer_t* buf);
  int (*release)(void* context);
  int (*copy_to_host)(void* context, struct cinn_buffer_t* buf);
  int (*copy_to_device)(void* context, struct cinn_buffer_t* buf);
  int (*buffer_copy)(void* context, struct cinn_buffer_t* src, struct cinn_buffer_t* dst);
};

// The device implementations
extern cinn_device_interface_t cinn_x86_device_interface;

inline float cinn_buffer_load_float32(struct cinn_buffer_t* buf, uint32_t index) {
  return ((float*)buf->host_memory)[index];
}
inline double cinn_buffer_load_float64(struct cinn_buffer_t* buf, uint32_t index) {
  return ((double*)buf->host_memory)[index];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#define CINN_NOT_IMPLEMENTED           \
  fprintf(stderr, "Not Implemented!"); \
  abort();

#define ASSERT_NOT_NULL(v__)          \
  if (!v__) {                         \
    fprintf(stderr, #v__ " is null"); \
    return -1;                        \
  }
#define CINN_LOG(fmt, ...)                                                          \
  do {                                                                              \
    fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  } while (0)

#define CINN_CHECK(cond)                \
  if (!(cond)) {                        \
    CINN_LOG("check %s failed", #cond); \
    abort();                            \
  }
#define CINN_CHECKP(cond, ...) \
  if (!(cond)) {               \
    CINN_LOG(__VA_ARGS__);     \
    abort();                   \
  }

#endif  // CINN_RUNTIME_CINN_RUNTIME_H_

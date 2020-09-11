#pragma once

#if !defined(NDEBUG)
#define CINN_DEBUG
#endif

#define CINN_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete

#ifndef CINN_NOT_IMPLEMENTED
#define CINN_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented";
#endif

#define CINN_RESULT_SHOULD_USE __attribute__((warn_unused_result))

/**
 * A trick to enforce the registry.
 *
 * usage:
 *
 * CINN_REGISTER_HELPER(some_key) {
 *   // register methods
 * }
 *
 * CINN_USE_REGISTER(some_key);
 */
#define CINN_REGISTER_HELPER(symbol__) bool __cinn__##symbol__##__registrar()
#define CINN_USE_REGISTER(symbol__)              \
  extern bool __cinn__##symbol__##__registrar(); \
  [[maybe_unused]] static bool __cinn_extern_registrar_##symbol__ = __cinn__##symbol__##__registrar();

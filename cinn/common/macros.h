#pragma once

#if !defined(NDEBUG)
#define CINN_DEBUG
#endif

#define CINN_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete

#define CINN_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented";

#define CINN_RESULT_SHOULD_USE __attribute__((warn_unused_result))

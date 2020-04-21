#pragma once

#if !defined(NDEBUG)
#define CINN_DEBUG
#endif

#define CINN_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete

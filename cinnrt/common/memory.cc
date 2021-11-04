#include "infrt/common/memory.h"

namespace infrt {

using infrt::common::Target;

namespace {

class X86MemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override { return ::malloc(nbytes); }
  void free(void* data) override {
    if (!data) return;
    ::free(data);
  }
  void* aligned_alloc(size_t alignment, size_t nbytes) override { return ::aligned_alloc(alignment, nbytes); }
};

}  // namespace

MemoryManager::MemoryManager() {
  Register(Target::Arch::Unk, new X86MemoryMng);
  Register(Target::Arch::X86, new X86MemoryMng);
}

}  // namespace infrt

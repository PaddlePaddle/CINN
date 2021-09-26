#pragma once

#include <glog/logging.h>

#include <memory>
#include <unordered_map>

#include "cinn/common/macros.h"
#include "cinn/common/target.h"

namespace cinn {
namespace hlir {
namespace framework {

class MemoryInterface {
 public:
  virtual void* malloc(size_t nbytes) = 0;
  virtual void free(void* data)       = 0;
  virtual void* aligned_alloc(size_t alignment, size_t nbytes) { return nullptr; }
  virtual ~MemoryInterface() {}
};

/**
 * MemoryManager holds a map of MemoryInterface for each architecture.
 */
class MemoryManager final {
 public:
  using key_t = common::Target::Arch;

  static MemoryManager& Global() {
    static auto* x = new MemoryManager;
    return *x;
  }

  MemoryInterface* Retrieve(key_t key) CINN_RESULT_SHOULD_USE {
    auto it = memory_mngs_.find(key);
    if (it != memory_mngs_.end()) return it->second.get();
    return nullptr;
  }

  MemoryInterface* RetrieveSafely(key_t key) {
    auto* res = Retrieve(key);
    CHECK(res) << "no MemoryInterface for architecture [" << key << "]";
    return res;
  }

  MemoryInterface* Register(key_t key, MemoryInterface* item) {
    CHECK(!memory_mngs_.count(key)) << "Duplicate register [" << key << "]";
    memory_mngs_[key].reset(item);
    return item;
  }

 private:
  MemoryManager();

  std::unordered_map<common::Target::Arch, std::unique_ptr<MemoryInterface>> memory_mngs_;

  CINN_DISALLOW_COPY_AND_ASSIGN(MemoryManager);
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

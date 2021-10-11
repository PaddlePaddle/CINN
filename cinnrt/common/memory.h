#pragma once

#include <glog/logging.h>

#include <absl/container/flat_hash_map.h>
#include <memory>

#include "cinnrt/common/macros.h"
#include "cinnrt/common/target.h"

namespace cinnrt {

class MemoryInterface {
 public:
  virtual void* malloc(size_t nbytes) = 0;
  virtual void free(void* data)       = 0;
  virtual void* aligned_alloc(size_t alignment, size_t nbytes) { return nullptr; }
  virtual ~MemoryInterface() {}
};

/**
 * MemoryManager holds a map of MemoryInterface for each articture.
 */
class MemoryManager final {
 public:
  using key_t = cinnrt::common::Target::Arch;

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

  absl::flat_hash_map<cinnrt::common::Target::Arch, std::unique_ptr<MemoryInterface>> memory_mngs_;

  CINN_DISALLOW_COPY_AND_ASSIGN(MemoryManager);
};

}  // namespace cinnrt

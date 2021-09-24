#include "cinn/ir/registry.h"

#include <map>
#include <mutex>  // NOLINT

namespace cinn::ir {
struct Registry::Manager {
  static Manager *Global() {
    static Manager manager;
    return &manager;
  }

  std::mutex mu;
  std::map<std::string, Registry *> functions;

 private:
  Manager()                = default;
  Manager(const Manager &) = delete;
  void operator=(Manager &) = delete;
};

Registry &Registry::SetBody(lang::PackedFunc f) {
  func_ = f;
  return *this;
}

Registry &Registry::SetBody(lang::PackedFunc::body_t f) {
  func_ = lang::PackedFunc(f);
  return *this;
}

Registry::Registry(const std::string &name) : name_(name) {}

/*static*/ Registry &Registry::Register(const std::string &name, bool can_override) {
  auto *manager = Registry::Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  if (manager->functions.count(name)) {
    CHECK(can_override) << "Global PackedFunc[" << name << "] is already exists";
  }

  auto *r                  = new Registry(name);
  manager->functions[name] = r;
  return *r;
}

/*static*/ bool Registry::Remove(const std::string &name) {
  auto manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);

  if (auto it = manager->functions.find(name); it != manager->functions.end()) {
    manager->functions.erase(it);
    return true;
  }
  return false;
}

/*static*/ const lang::PackedFunc *Registry::Get(const std::string &name) {
  auto *manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  auto *r = manager->functions[name];
  if (r) {
    return &r->func_;
  }
  return nullptr;
}

/*static*/ std::vector<std::string> Registry::ListNames() {
  auto *manager = Manager::Global();
  std::lock_guard<std::mutex> lock(manager->mu);
  std::vector<std::string> keys;
  for (const auto &_k_v_ : manager->functions) {
    auto &k = std::get<0>(_k_v_);
    auto &v = std::get<1>(_k_v_);
    keys.push_back(k);
  }
  return keys;
}

}  // namespace cinn::ir

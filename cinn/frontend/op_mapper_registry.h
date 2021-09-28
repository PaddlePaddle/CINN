#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "cinn/common/macros.h"
#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/utils/registry.h"

namespace cinn {
namespace frontend {

class OpMapperContext {
 public:
  OpMapperContext(hlir::framework::Scope* scope,
                  const common::Target& target,
                  NetBuilder* builder,
                  absl::flat_hash_map<std::string, Variable>* var_map,
                  absl::flat_hash_map<std::string, std::string>* var_model_to_program_map)
      : scope_(scope),
        target_(target),
        builder_(builder),
        var_map_(var_map),
        var_model_to_program_map_(var_model_to_program_map) {}

  hlir::framework::Scope* scope_{nullptr};
  const common::Target& target_;
  NetBuilder* builder_{nullptr};
  absl::flat_hash_map<std::string, Variable>* var_map_{nullptr};
  // map from var in Paddle model to var name in program.
  absl::flat_hash_map<std::string, std::string>* var_model_to_program_map_{nullptr};
};

class OpMapperKernel {
 public:
  using KernelFunc = std::function<void(const paddle::cpp::OpDesc&, const OpMapperContext&)>;

  OpMapperKernel() = default;

  OpMapperKernel& Set(const KernelFunc& kernel) {
    kernel_ = kernel;
    return *this;
  }
  void Run(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) const { kernel_(op_desc, ctx); }

  std::string name;

 private:
  KernelFunc kernel_;
};

class OpMapperRegistry : public Registry<OpMapperKernel> {
 public:
  OpMapperRegistry() = default;

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(OpMapperRegistry);
};

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
  struct __test_global_namespace_##uniq_name##__ {};   \
  static_assert(                                       \
      std::is_same<::__test_global_namespace_##uniq_name##__, __test_global_namespace_##uniq_name##__>::value, msg)

#define UNIQUE_OPMAPPER_NAME(OpName) static ::cinn::frontend::OpMapperKernel& __op_mapper_registrar_##OpName

#define CINN_REGISTER_OPMAPPER(OpName, Kernel)                 \
  CINN_STR_CONCAT(UNIQUE_OPMAPPER_NAME(OpName), __COUNTER__) = \
      ::cinn::frontend::OpMapperRegistry::Global()->__REGISTER_OR_GET__(#OpName).Set(Kernel);

}  // namespace frontend
}  // namespace cinn

// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/ir/schedule_desc.h"

#include <glog/logging.h>

#include <utility>

#include "cinn/common/macros.h"
#include "cinn/utils/registry.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

class PackedStepContext;
using StepApplyFunc = std::vector<Expr> (*)(PackedStepContext*);

class StepKindInfo {
 public:
  // Setter
  // format: {"<name1>", "<name2>", ...}
  StepKindInfo& Inputs(std::vector<std::string>&& inputs) {
    inputs_ = inputs;
    return *this;
  }
  //// format: {"<name1>", "<name2>", ...}
  // StepKindInfo& Outputs(std::vector<std::string>&& outputs) {
  //    outputs_ = outputs;
  //    return *this;
  //}
  // format: {"<name1>", "<name2>", ...}
  // ? format: {"<name1>:<type1>", "<name1>:<type1>", ...}
  StepKindInfo& Attrs(std::vector<std::string>&& attrs) {
    attrs_ = attrs;
    return *this;
  }
  // format: CINN_STEP_FUNC(...)
  StepKindInfo& SetApplyFn(StepApplyFunc&& func) {
    apply_func_ = func;
    return *this;
  }

 private:
  friend class PackedStepContext;

  std::vector<std::string> inputs_;
  // std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
  StepApplyFunc apply_func_{nullptr};
};

class StepKindRegistry : public Registry<StepKindInfo> {
 public:
  StepKindRegistry() = default;

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(StepKindRegistry);
};

#define CINN_BUILD_STEP_KIND(TypeName)                                \
  static ::cinn::ir::StepKindInfo& __step_kind_registrar_##TypeName = \
      ::cinn::ir::StepKindRegistry::Global()->__REGISTER_OR_GET__(#TypeName)

class PackedStepContext {
 public:
  explicit PackedStepContext(IRSchedule* schedule) : ir_schedule_(schedule) {}

  void Initialize(const ScheduleDesc::Step& desc) {
    CHECK(!desc.type.empty()) << "Name of StepKind is empty";
    const StepKindInfo* step_kind_ = StepKindRegistry::Global()->Find(desc.type);
    CHECK(step_kind_) << "Can't find StepKind:" << desc.type;

    // build inputs
    size_t input_idx = 0;
    for (auto&& param_name : step_kind_->inputs_) {
      auto arg_it = desc.inputs.find(param_name);
      CHECK(arg_it != desc.inputs.end()) << "Can't find param:" << param_name;
      const std::vector<std::string>& arguments = arg_it->second;
      CHECK(!arguments.empty()) << "No arguments:" << param_name;
      for (auto&& arg_name : arguments) {
        auto expr_it = exprs.find(arg_name);
        CHECK(expr_it != exprs.end()) << "Can't find argument:" << arg_name;
        inputs_.emplace_back(expr_it->second);
      }
      input_range_.emplace_back(input_idx, input_idx + arguments.size());
      input_idx += arguments.size();
    }

    // build attrs
    size_t attr_idx = 0;
  }

  IRSchedule* ScheduleHandler() const { return ir_schedule_; }

  Expr InputAt(size_t idx) const {
    CHECK_LT(idx, input_range_.size()) << "idx overranges";
    const auto& range = input_range_.at(idx);
    CHECK(range.second - range.first == 1) << "not single param";
    return inputs_[range_.first];
  }

  std::vector<Expr> InputsAt(size_t idx) const {
    CHECK_LT(idx, input_range_.size()) << "idx overranges";
    const auto& range = input_range_.at(idx);
    std::vector<Expr> results;
    for (size_t s = range.first; s < range.second; ++s) {
      results.emplace_back(inputs_[s])
    }
    return results;
  }

  template <typename DataType>
  DataType AttrAt(size_t idx) const;

 private:
  IRSchedule* ir_schedule_;
  std::vector<Expr> inputs_;
  std::vector<std::pair<size_t, size_t>> input_range_;
  std::vector<utils::Attribute> attrs_;
};

#define CINN_SPECIALIZE_ApplyCallHelper(attr_type)                                         \
  template <typename... Tail>                                                              \
  struct ApplyCallHelper<attr_type, Tail...> {                                             \
    template <int in_idx, int attr_idx, typename... PreviousArgs>                          \
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) { \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                                    \
      ApplyCallHelper<Tail...>::template Apply<in_idx, attr_idx + 1>(ctx, pargs..., arg);  \
    }                                                                                      \
  }

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct ApplyFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct ApplyFuncImpl<Return (*)(Args...), impl_fn> {
  static std::vector<Expr> Apply(PackedStepContext* ctx) {
    ApplyCallHelper<Args..., TypeTag<int>>::template Apply<0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct ApplyCallHelper;

  // first argument must be IRSchedule
  template <typename... Tail>
  struct ApplyCallHelper<IRSchedule*, Tail...> {
    template <int in_idx, int attr_idx>
    static std::vector<Expr> Apply(PackedStepContext* ctx) {
      static_assert(in_idx == 0, "First argument must be IRSchedule*");
      IRSchedule* ir_schedule = ctx->ScheduleHandler();
      ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx>(ctx, ir_schedule);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const Expr&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      auto& arg = ctx->InputAt(in_idx);
      ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx>(ctx, pargs..., arg);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const std::vector<Expr>&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      auto& arg = ctx->InputsAt(in_idx);
      ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx>(ctx, pargs..., arg);
    }
  };

  CINN_SPECIALIZE_ApplyCallHelper(bool);
  CINN_SPECIALIZE_ApplyCallHelper(int);
  CINN_SPECIALIZE_ApplyCallHelper(float);
  CINN_SPECIALIZE_ApplyCallHelper(const std::string&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<bool>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<int>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<float>&);
  CINN_SPECIALIZE_ApplyCallHelper(const std::vector<std::string>&);

  template <typename T>
  struct ApplyReturnHelper;

  template <>
  struct ApplyReturnHelper<void> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) {
      impl_fn(args...);
      return {};
    }
  };

  template <>
  struct ApplyReturnHelper<Expr> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) {
      auto ret = impl_fn(args...);
      return {ret};
    }
  };

  template <>
  struct ApplyReturnHelper<std::vector<Expr>> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) { return impl_fn(args...); }
  };

  // end: base template
  template <typename T>
  struct ApplyCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      return ApplyReturnHelper<Return>::Apply(ctx, pargs...);
    }
  };
};

#define CINN_STEP_KIND_APPLY_FUNC(...) ::cinn::ir::ApplyFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Apply

// implement demo step kind

// ---- ScheduleDesc
ScheduleDesc::ScheduleDesc(const proto::ScheduleDesc& desc_proto) {}

void ScheduleDesc::Append(Step&& step) {}

void ScheduleDesc::Replay(IRSchedule* schedule) {}

proto::ScheduleDesc ScheduleDesc::ToProto() const {}

}  // namespace ir
}  // namespace cinn

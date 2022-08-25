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
  // format: {"<name1>", "<name2>", ...}
  StepKindInfo& Inputs(std::vector<std::string>&& inputs) {
    inputs_ = inputs;
    return *this;
  }
  // format: {"<name1>", "<name2>", ...}
  StepKindInfo& Attrs(std::vector<std::string>&& attrs) {
    attrs_ = attrs;
    return *this;
  }
  // format: CINN_STEP_FUNC(...)
  StepKindInfo& SetApplyFn(StepApplyFunc&& func) {
    apply_func_ = func;
    return *this;
  }

  // compatible for Registry::EntryType
  std::string name;

 private:
  friend class PackedStepContext;

  std::vector<std::string> inputs_;
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

  void Build(const ScheduleDesc::Step& desc) {
    CHECK(!desc.type.empty()) << "Name of StepKind is empty";
    const StepKindInfo* step_kind_ = StepKindRegistry::Global()->Find(desc.type);
    CHECK(step_kind_) << "Can't find StepKind:" << desc.type;

    // build inputs
    size_t input_idx = 0;
    for (auto&& param_name : step_kind_->inputs_) {
      auto arg_it = desc.inputs.find(param_name);
      CHECK(arg_it != desc.inputs.end()) << "Can't find param:" << param_name;
      auto&& args = arg_it->second;
      inputs_.insert(inputs_.end(), std::make_move_iterator(args.begin()), std::make_move_iterator(args.end()));
      input_range_.emplace_back(input_idx, input_idx + args.size());
      input_idx += args.size();
    }

    // build attrs
    size_t attr_idx = 0;
    for (auto&& attr_name : step_kind_->attrs_) {
      auto attr_it = desc.attrs.find(attr_name);
      CHECK(attr_it != desc.attrs.end()) << "Can't find attribute:" << attr_name;
      attrs_.emplace_back(attr_it->second);
      ++attr_idx;
    }
  }

  IRSchedule* ScheduleHandler() const { return ir_schedule_; }

  Expr InputAt(size_t idx) const {
    CHECK_LT(idx, input_range_.size()) << "idx overranges";
    const auto& range = input_range_.at(idx);
    CHECK(range.second - range.first == 1) << "not single param";
    return inputs_[range.first];
  }

  std::vector<Expr> InputsAt(size_t idx) const {
    CHECK_LT(idx, input_range_.size()) << "idx overranges";
    const auto& range = input_range_.at(idx);
    std::vector<Expr> results;
    for (size_t s = range.first; s < range.second; ++s) {
      results.emplace_back(inputs_[s]);
    }
    return results;
  }

  template <typename AttrType>
  const AttrType& AttrAt(size_t idx) const {
    try {
      return absl::get<AttrType>(attrs_.at(idx));
    } catch (absl::bad_variant_access& ex) {
      LOG(FATAL) << "Attribute cast error:" << ex.what();
      throw ex;
    }
  }

 private:
  IRSchedule* ir_schedule_;
  std::vector<Expr> inputs_;
  std::vector<std::pair<size_t, size_t>> input_range_;
  std::vector<utils::Attribute> attrs_;
};

#define CINN_SPECIALIZE_ApplyCallHelper(attr_type)                                                        \
  template <typename... Tail>                                                                             \
  struct ApplyCallHelper<attr_type, Tail...> {                                                            \
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>                            \
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {                \
      using rf_attr_type = std::remove_reference<attr_type>::type;                                        \
      using rc_attr_type = std::remove_const<rf_attr_type>::type;                                         \
      const auto& arg    = ctx->AttrAt<rc_attr_type>(attr_idx);                                           \
      return ApplyCallHelper<Tail...>::template Apply<in_idx, attr_idx + 1, out_idx>(ctx, pargs..., arg); \
    }                                                                                                     \
  }

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct ApplyFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct ApplyFuncImpl<Return (*)(Args...), impl_fn> {
  static std::vector<Expr> Apply(PackedStepContext* ctx) {
    return ApplyCallHelper<Args..., TypeTag<int>>::template Apply<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct ApplyCallHelper;

  // first argument must be IRSchedule
  template <typename... Tail>
  struct ApplyCallHelper<IRSchedule*, Tail...> {
    template <int in_idx, int attr_idx, int out_idx>
    static std::vector<Expr> Apply(PackedStepContext* ctx) {
      static_assert(in_idx == 0, "IRSchedule* must be the first argument");
      IRSchedule* ir_schedule = ctx->ScheduleHandler();
      return ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx, out_idx>(ctx, ir_schedule);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const Expr&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      static_assert(in_idx > 0, "Other arguments should be right behind IRSchedule*");
      auto arg = ctx->InputAt(in_idx);
      return ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx, out_idx>(ctx, pargs..., arg);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const std::vector<Expr>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      static_assert(in_idx > 0, "Other arguments should be right behind IRSchedule*");
      auto arg = ctx->InputsAt(in_idx);
      return ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx, out_idx>(ctx, pargs..., arg);
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

  template <int out_idx, typename T>
  struct ApplyReturnHelper;

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, void> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) {
      impl_fn(args...);
      return {};
    }
  };

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, Expr> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) {
      auto ret = impl_fn(args...);
      return {ret};
    }
  };

  template <int out_idx>
  struct ApplyReturnHelper<out_idx, std::vector<Expr>> {
    static std::vector<Expr> Apply(PackedStepContext* ctx, const Args&... args) { return impl_fn(args...); }
  };

  // end: base template
  template <typename T>
  struct ApplyCallHelper<TypeTag<T>> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      static_assert(out_idx == 0, "Output is exported from return value");
      return ApplyReturnHelper<out_idx, Return>::Apply(ctx, pargs...);
    }
  };
};

#define CINN_STEP_KIND_APPLY_FUNC(...) ::cinn::ir::ApplyFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Apply

// implement demo step kind
std::vector<Expr> GetLoops(IRSchedule* sch, const Expr& block) { return sch->GetLoops(block); }

std::vector<Expr> GetLoopsWithName(IRSchedule* sch, const std::string& block_name) { return sch->GetLoops(block_name); }

std::vector<Expr> GetAllBlocks(IRSchedule* sch) { return sch->GetAllBlocks(); }

Expr GetBlock(IRSchedule* sch, const std::string& block_name) { return sch->GetBlock(block_name); }

CINN_BUILD_STEP_KIND(GetLoops).Inputs({"block"}).SetApplyFn(CINN_STEP_KIND_APPLY_FUNC(GetLoops));

CINN_BUILD_STEP_KIND(GetLoopsWithName).Attrs({"block_name"}).SetApplyFn(CINN_STEP_KIND_APPLY_FUNC(GetLoopsWithName));

CINN_BUILD_STEP_KIND(GetAllBlocks).SetApplyFn(CINN_STEP_KIND_APPLY_FUNC(GetAllBlocks));

CINN_BUILD_STEP_KIND(GetBlock).Attrs({"block_name"}).SetApplyFn(CINN_STEP_KIND_APPLY_FUNC(GetBlock));

// ---- ScheduleDesc

// std::vector<Expr> TranslateInputExprs(const std::vector<std::string>& expr_names);
// void TranslateAddOutputExprs(const std::vector<Exprs>& exprs);
/*! \brief ObjectRef hash functor */
// struct ExprPtrHash {
//  size_t operator()(const Expr& e) const { return operator()(e.data_); }
//
//  template <typename T>
//  size_t operator()(const ObjectPtr<T>& a) const {
//    return std::hash<Object*>()(a.get());
//  }
//};
//
///*! \brief ObjectRef equal functor */
// struct ObjectPtrEqual {
//  bool operator()(const ObjectRef& a, const ObjectRef& b) const { return a.same_as(b); }
//
//  template <typename T>
//  size_t operator()(const ObjectPtr<T>& a, const ObjectPtr<T>& b) const {
//    return a == b;
//  }
//};

void ScheduleDesc::Append(Step&& step) { steps.emplace_back(step); }

void ScheduleDesc::Replay(IRSchedule* schedule) { return ReplayWithProto(this->ToProto(), schedule); }

proto::ScheduleDesc ScheduleDesc::ToProto() const {
  proto::ScheduleDesc sch_desc;
  absl::flat_hash_map<Expr, std::string> expr2name;
  /*
    for (auto&& step : steps) {
        auto* step_desc = sch_desc.add_steps();
        step_desc->set_type(step.type);
        for (auto&& param2exprs : step.inputs) {
            const std::string& param_name = param2exprs.first;
            auto* expr_desc = step_desc->add_inputs();
            expr_desc->set_parameter(param_name);
            for (auto&& expr : param2exprs.second) {
                auto expr_it = expr2name.find(expr);
                CHECK(expr_it != expr2name.end()) << "Can't find expr of param_name: "
                    << param_name << ", expr:" << expr;
                expr_desc->add_arguments(expr_it->second);
            }
        }

        for (auto&& expr : step.outputs) {
            auto* expr_desc = step_desc->add_outputs();
            // parameter is empty
            std::string local_name = "e" + std::to_string(expr2name.size());
            expr2name.emplace(expr, local_name);
            expr_desc->add_arguments(expr2name.at(expr));
        }

        for (auto&& attr2value : step.attrs) {
            auto* attr_desc = step_desc->add_attrs();
            attr_desc->set_name(attr2value.first);
            // todo
            attr_desc->set_dtype(proto::ScheduleDesc_Attr_DataType_BOOLEAN);
            attr_desc->set_b(false);
        }
    }
    */
  return sch_desc;
}

void ScheduleDesc::ReplayWithProto(const proto::ScheduleDesc& desc_proto, IRSchedule* sch) {
  absl::flat_hash_map<std::string, Expr> name2expr;
}

}  // namespace ir
}  // namespace cinn

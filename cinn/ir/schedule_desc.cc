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

#include <typeinfo>
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

  std::vector<Expr> Apply(PackedStepContext* context) const { return apply_func_(context); }

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
  explicit PackedStepContext(const ScheduleDesc::Step& desc, const StepKindInfo* step_kind, IRSchedule* schedule)
      : ir_schedule_(schedule) {
    Build(desc, step_kind);
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
      LOG(FATAL) << "Attribute cast error, idx:" << idx << ", get tpye:" << typeid(AttrType).name()
                 << ", real index:" << attrs_.at(idx).index();
      throw ex;
    }
  }

 private:
  void Build(const ScheduleDesc::Step& desc, const StepKindInfo* step_kind) {
    // build inputs
    size_t input_idx = 0;
    for (auto&& param_name : step_kind->inputs_) {
      auto arg_it = desc.inputs.find(param_name);
      CHECK(arg_it != desc.inputs.end()) << "Can't find param:" << param_name;
      auto&& args = arg_it->second;
      inputs_.insert(inputs_.end(), std::make_move_iterator(args.begin()), std::make_move_iterator(args.end()));
      input_range_.emplace_back(input_idx, input_idx + args.size());
      input_idx += args.size();
    }

    // build attrs
    size_t attr_idx = 0;
    for (auto&& attr_name : step_kind->attrs_) {
      auto attr_it = desc.attrs.find(attr_name);
      CHECK(attr_it != desc.attrs.end()) << "Can't find attribute:" << attr_name;
      attrs_.emplace_back(attr_it->second);
      ++attr_idx;
    }
  }

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
      auto arg = ctx->InputAt(in_idx - 1);
      return ApplyCallHelper<Tail...>::template Apply<in_idx + 1, attr_idx, out_idx>(ctx, pargs..., arg);
    }
  };

  template <typename... Tail>
  struct ApplyCallHelper<const std::vector<Expr>&, Tail...> {
    template <int in_idx, int attr_idx, int out_idx, typename... PreviousArgs>
    static std::vector<Expr> Apply(PackedStepContext* ctx, const PreviousArgs&... pargs) {
      static_assert(in_idx > 0, "Other arguments should be right behind IRSchedule*");
      auto arg = ctx->InputsAt(in_idx - 1);
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

#define STEP_APPLY_FUNC_WARPPER(...) ::cinn::ir::ApplyFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Apply

// implement demo step kind
// clang-format off
std::vector<Expr> GetLoops(IRSchedule* sch, const Expr& block) {
    return sch->GetLoops(block);
}
CINN_BUILD_STEP_KIND(GetLoops)
    .Inputs({"block"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(GetLoops));

std::vector<Expr> GetLoopsWithName(IRSchedule* sch, const std::string& block_name) {
    return sch->GetLoops(block_name);
}
CINN_BUILD_STEP_KIND(GetLoopsWithName)
    .Attrs({"block_name"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(GetLoopsWithName));

std::vector<Expr> GetAllBlocks(IRSchedule* sch) {
    return sch->GetAllBlocks();
}
CINN_BUILD_STEP_KIND(GetAllBlocks)
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(GetAllBlocks));

Expr GetBlock(IRSchedule* sch, const std::string& block_name) {
    return sch->GetBlock(block_name);
}
CINN_BUILD_STEP_KIND(GetBlock)
    .Attrs({"block_name"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(GetBlock));

std::vector<Expr> Split(IRSchedule* sch, const Expr& loop, const std::vector<int>& factors) {
    return sch->Split(loop, factors);
}
CINN_BUILD_STEP_KIND(Split)
    .Inputs({"loop"})
    .Attrs({"factors"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(Split));

std::vector<Expr> SplitWithName(IRSchedule* sch,
                        const std::string& block_name,
                        int loop_index,
                        const std::vector<int>& factors) {
    return sch->Split(block_name, loop_index, factors);
}
CINN_BUILD_STEP_KIND(SplitWithName)
    .Attrs({"block_name", "loop_index", "factors"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(SplitWithName));

Expr Fuse(IRSchedule* sch, const std::vector<Expr>& loops) {
    return sch->Fuse(loops);
}
CINN_BUILD_STEP_KIND(Fuse)
    .Inputs({"loops"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(Fuse));

Expr FuseWithBlockName(IRSchedule* sch, const std::string& block_name,
          const std::vector<int>& loops_index) {
    return sch->Fuse(block_name, loops_index);
}
CINN_BUILD_STEP_KIND(FuseWithBlockName)
    .Attrs({"block_name", "loops_index"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(FuseWithBlockName));

Expr FuseWithBlock(IRSchedule* sch, const Expr& block,
                   const std::vector<int>& loops_index) {
    return sch->Fuse(block, loops_index);
}
CINN_BUILD_STEP_KIND(FuseWithBlock)
    .Inputs({"block"})
    .Attrs({"loops_index"})
    .SetApplyFn(STEP_APPLY_FUNC_WARPPER(FuseWithBlock));

/*
void ComputeAt(IRSchedule* sch, const Expr& block, const Expr& loop) {

}
void SimpleComputeAt(IRSchedule* sch, const Expr& block, const Expr& loop) {

}
Expr GetRootBlock(IRSchedule* sch, const Expr& expr) {

}
Expr CacheRead(IRSchedule* sch, const Expr& block, int read_buffer_index, const std::string& memory_type) {

}
Expr CacheWrite(IRSchedule* sch, const Expr& block, int write_buffer_index, const std::string& memory_type) {

}
void SyncThreads(IRSchedule* sch, const Expr& ir_node, bool after_node = true) {

}
void SetBuffer(IRSchedule* sch, Expr& block, const std::string& memory_type, bool fixed = false) {

}

void Reorder(IRSchedule* sch, const std::vector<Expr>& loops) {

}
void ReorderWithBlockName(IRSchedule* sch, const std::string& block_name, const std::vector<int>& loops_index) {

}

void ReorderWithBlock(IRSchedule* sch, const Expr& block, const std::vector<int>& loops_index) {
}

void Parallel(IRSchedule* sch, const Expr& loop) {

}
void Vectorize(IRSchedule* sch, const Expr& loop, int factor) {

}
void Unroll(IRSchedule* sch, const Expr& loop) {

}
void ComputeInline(IRSchedule* sch, const Expr& schedule_block) {

}
void Bind(IRSchedule* sch, const Expr& loop, const std::string& thread_axis) {

}
Expr Rfactor(IRSchedule* sch, const Expr& rf_loop, int rf_axis) {
    return sch->Rfactor(rf_loop, rf_axis);
}
void MergeExprs(IRSchedule* sch) {
    return sch->MergeExprs();
}
*/

// clang-format on

// ---- ScheduleDesc
void AttrVariantToProto(const utils::Attribute& attr, proto::ScheduleDesc_Attr* attr_proto) {
#define SET_DESC_SINGLE_ITEM(index, built_type, proto_type, proto_field)   \
  case index:                                                              \
    attr_proto->set_dtype(proto::ScheduleDesc_Attr_DataType_##proto_type); \
    attr_proto->set_##proto_field(absl::get<built_type>(attr));            \
    break;

#define SET_DESC_REPEATED_ITEM(index, built_type, proto_type, proto_field) \
  case index: {                                                            \
    attr_proto->set_dtype(proto::ScheduleDesc_Attr_DataType_##proto_type); \
    const auto& values = absl::get<built_type>(attr);                      \
    attr_proto->mutable_##proto_field()->Reserve(values.size());           \
    *attr_proto->mutable_##proto_field() = {values.begin(), values.end()}; \
    break;                                                                 \
  }

  switch (attr.index()) {
    SET_DESC_SINGLE_ITEM(0, bool, BOOLEAN, b);
    SET_DESC_SINGLE_ITEM(1, float, FLOAT, f);
    SET_DESC_SINGLE_ITEM(2, int, INT, i);
    SET_DESC_SINGLE_ITEM(3, std::string, STRING, s);
    SET_DESC_REPEATED_ITEM(4, std::vector<bool>, BOOLEANS, bools);
    SET_DESC_REPEATED_ITEM(5, std::vector<int>, INTS, ints);
    SET_DESC_REPEATED_ITEM(6, std::vector<float>, FLOATS, floats);
    SET_DESC_REPEATED_ITEM(7, std::vector<std::string>, STRINGS, strings);
    default:
      LOG(FATAL) << "Invalid index:" << attr.index();
  }

#undef SET_DESC_SINGLE_ITEM
#undef SET_DESC_REPEATED_ITEM
}

utils::Attribute AttrProtoToVariant(const proto::ScheduleDesc_Attr& attr) {
  utils::Attribute value;
#define PARSE_DESC_SINGLE_ITEM(proto_type, proto_field, built_type) \
  case proto::ScheduleDesc_Attr_DataType_##proto_type:              \
    value = built_type(attr.proto_field());                         \
    break;

#define PARSE_DESC_REPEATED_ITEM(proto_type, proto_field, built_type)           \
  case proto::ScheduleDesc_Attr_DataType_##proto_type:                          \
    value = built_type({attr.proto_field().begin(), attr.proto_field().end()}); \
    break;

  switch (attr.dtype()) {
    PARSE_DESC_SINGLE_ITEM(BOOLEAN, b, bool);
    PARSE_DESC_SINGLE_ITEM(INT, i, int);
    PARSE_DESC_SINGLE_ITEM(FLOAT, f, float);
    PARSE_DESC_SINGLE_ITEM(STRING, s, std::string);
    PARSE_DESC_REPEATED_ITEM(BOOLEANS, bools, std::vector<bool>);
    PARSE_DESC_REPEATED_ITEM(INTS, ints, std::vector<int>);
    PARSE_DESC_REPEATED_ITEM(FLOATS, floats, std::vector<float>);
    PARSE_DESC_REPEATED_ITEM(STRINGS, strings, std::vector<std::string>);
    default:
      LOG(FATAL) << "Invalid type:" << attr.DebugString();
  }

#undef PARSE_DESC_SINGLE_ITEM
#undef PARSE_DESC_REPEATED_ITEM
  return value;
}

struct ExprHash {
  size_t operator()(const Expr& e) const { return std::hash<IrNode*>()(e.ptr()); }
};
// Expr equal functor
struct ExprEqual {
  bool operator()(const Expr& lhs, const Expr& rhs) const { return lhs.get() == rhs.get(); }
};

void ScheduleDesc::Append(Step&& step) { steps.emplace_back(step); }

void ScheduleDesc::Replay(IRSchedule* schedule) { return ReplayWithProto(this->ToProto(), schedule); }

proto::ScheduleDesc ScheduleDesc::ToProto() const {
  proto::ScheduleDesc desc_proto;
  absl::flat_hash_map<Expr, std::string, ExprHash, ExprEqual> expr2name;
  for (auto&& step : steps) {
    auto* step_proto = desc_proto.add_steps();
    step_proto->set_type(step.type);
    for (auto&& param2exprs : step.inputs) {
      const std::string& param_name = param2exprs.first;
      auto* expr_desc               = step_proto->add_inputs();
      expr_desc->set_parameter(param_name);
      for (auto&& expr : param2exprs.second) {
        auto expr_it = expr2name.find(expr);
        CHECK(expr_it != expr2name.end()) << "Can't find expr of param_name: " << param_name;
        expr_desc->add_arguments(expr_it->second);
      }
    }

    for (auto&& expr : step.outputs) {
      std::string local_name = "e" + std::to_string(expr2name.size());
      expr2name.emplace(expr, local_name);
      step_proto->add_outputs(expr2name.at(expr));
    }

    for (auto&& attr2value : step.attrs) {
      auto* attr_proto       = step_proto->add_attrs();
      const auto& attr_value = attr2value.second;
      VLOG(5) << "Attr.index:" << attr_value.index();
      attr_proto->set_name(attr2value.first);
      AttrVariantToProto(attr_value, attr_proto);
    }
  }
  return desc_proto;
}

void ScheduleDesc::ReplayWithProto(const proto::ScheduleDesc& desc_proto, IRSchedule* sch) {
  // VLOG(4) << "proto::ScheduleDesc:\n" << desc_proto.DebugString();
  LOG(INFO) << "proto::ScheduleDesc:\n" << desc_proto.DebugString();
  if (desc_proto.steps().empty()) {
    LOG(WARNING) << "Input proto::ScheduleDesc is empty";
    return;
  }

  absl::flat_hash_map<std::string, Expr> name2expr;
  for (auto&& step_proto : desc_proto.steps()) {
    // VLOG(4) << "Replay step:\n" << step_proto.DebugString();
    LOG(INFO) << "Replay step:\n" << step_proto.DebugString();
    ScheduleDesc::Step step;
    step.type = step_proto.type();
    CHECK(!step.type.empty()) << "Name of StepKind is empty";
    const StepKindInfo* step_kind = StepKindRegistry::Global()->Find(step.type);
    CHECK(step_kind) << "Can't find StepKind:" << step.type;

    for (auto&& param2args : step_proto.inputs()) {
      for (auto&& arg : param2args.arguments()) {
        auto arg_it = name2expr.find(arg);
        CHECK(arg_it != name2expr.end()) << "Cant't find argument:" << arg;
        step.inputs[param2args.parameter()].emplace_back(arg_it->second);
      }
    }
    for (auto&& attr : step_proto.attrs()) {
      step.attrs[attr.name()] = AttrProtoToVariant(attr);
    }

    PackedStepContext context(step, step_kind, sch);
    step.outputs = step_kind->Apply(&context);
    CHECK_EQ(step_proto.outputs().size(), step.outputs.size()) << "Output size not matched";
    for (size_t i = 0; i < step.outputs.size(); ++i) {
      name2expr[step_proto.outputs(i)] = step.outputs.at(i);
    }
  }
}

}  // namespace ir
}  // namespace cinn

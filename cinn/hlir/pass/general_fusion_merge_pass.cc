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

#include <map>
#include <unordered_map>

#include "cinn/api/op_group_interface.h"
#include "cinn/common/is_reachable_predicator.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/pass/fusion_merge_pass_util.h"

DECLARE_bool(enhance_vertical_fusion_with_recompute);

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using OpGroupPtr  = std::shared_ptr<api::OpGroupInterface>;
using OpGroupList = std::vector<OpGroupPtr>;

using ConditionFunction = std::function<bool(const FusionHelperBase*, const GroupPtr&, const GroupPtr&)>;

class FuseHelper {
 public:
  virtual ~FuseHelper() = default;

  virtual bool AllOutputsSameSize(const OpGroupPtr& first, const OpGroupPtr& second) const = 0;

  virtual bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool HorizontalWithInjective(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool ElementwiseFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool BroadcastFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool InjectiveHorizontalWithReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseElementwise(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool ReduceFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool IsReachable(const OpGroupPtr& lhs, const OpGroupPtr& rhs) const = 0;

  virtual bool DetectCycleIfFuse(const OpGroupPtr& src, const OpGroupPtr& dst) const = 0;

  virtual bool IsConsumerSetsReachable(const OpGroupPtr& group,
                                       const std::unordered_set<OpGroupPtr>& consumers) const = 0;

 protected:
  FuseHelper() = default;
};

template <typename FusePassCtxT>
class GraphGroupFuseHelper final : public FuseHelper {
 public:
  explicit GraphGroupFuseHelper(const FusePassCtxT* ctx) : ctx_(ctx) {}

  bool AllOutputsSameSize(const OpGroupPtr& first, const OpGroupPtr& second) const override;

  bool HorizontalElementwiseFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool ElementwiseFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool HorizontalWithInjective(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool ElementwiseFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool BroadcastFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool InjectiveHorizontalWithReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool ReduceFuseElementwise(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool ReduceFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool ReduceFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const override;

  bool IsReachable(const OpGroupPtr& lhs, const OpGroupPtr& rhs) const override {
    return IsReachableInDag(lhs, rhs) || IsReachableInDag(rhs, lhs);
  }

  bool DetectCycleIfFuse(const OpGroupPtr& lhs, const OpGroupPtr& rhs) const override {
    return ReachableIfDirectEdgeIgnored(lhs, rhs) || ReachableIfDirectEdgeIgnored(rhs, lhs);
  }

  bool IsConsumerSetsReachable(const OpGroupPtr& group,
                               const std::unordered_set<OpGroupPtr>& consumers) const override {
    for (const auto& consumer : consumers) {
      if (group == consumer) {
        continue;
      }
      if (IsReachableInDag(consumer, group)) {
        return true;
      }
    }
    return false;
  }

 private:
  bool IsReachableInDag(const OpGroupPtr& producer, const OpGroupPtr& consumer) const {
    const auto& MinDepth4Node = [&](OpGroupPtr node) {
      return std::dynamic_pointer_cast<Graph::Group>(node)->min_depth;
    };
    const auto& MaxDepth4Node = [&](OpGroupPtr node) {
      return std::dynamic_pointer_cast<Graph::Group>(node)->max_depth;
    };
    const auto& VisitNextNodes = [&](OpGroupPtr node, const std::function<void(OpGroupPtr)>& Visit) {
      for (const auto& pair : node->producer2inputs()) {
        Visit(pair.first);
      }
    };
    common::IsReachablePredicator<OpGroupPtr> is_reachable(MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    return is_reachable(consumer, producer, [](OpGroupPtr) {});
  }

  bool ReachableIfDirectEdgeIgnored(const OpGroupPtr& producer, const OpGroupPtr& consumer) const {
    const auto& MinDepth4Node = [&](OpGroupPtr node) {
      return std::dynamic_pointer_cast<Graph::Group>(node)->min_depth;
    };
    const auto& MaxDepth4Node = [&](OpGroupPtr node) {
      return std::dynamic_pointer_cast<Graph::Group>(node)->max_depth;
    };
    const auto& VisitNextNodes = [&](OpGroupPtr node, const std::function<void(OpGroupPtr)>& Visit) {
      for (const auto& pair : node->producer2inputs()) {
        if (node == consumer && pair.first == producer) {
          continue;
        }
        Visit(pair.first);
      }
    };
    common::IsReachablePredicator<OpGroupPtr> is_reachable(MinDepth4Node, MaxDepth4Node, VisitNextNodes);
    return is_reachable(consumer, producer, [](OpGroupPtr) {});
  }

  const FusePassCtxT* ctx_;
};

class FusePassCtx {
 public:
  virtual ~FusePassCtx() {}

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void EnableFuse(const OpGroupPtr& first, const OpGroupPtr& second) = 0;

 protected:
  FusePassCtx() = default;
};

class LightwareFusePassCtx : public FusePassCtx {
 public:
  virtual ~LightwareFusePassCtx() {}

  virtual const OpGroupPtr& PickOpGroup() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void EnableFuse(const OpGroupPtr& first, const OpGroupPtr& second) = 0;

  virtual void EnableFuse(const OpGroupList& candidates) = 0;

 protected:
  LightwareFusePassCtx() = default;
};

class GraphGroupLightwareFusePassCtx final : public LightwareFusePassCtx {
 public:
  GraphGroupLightwareFusePassCtx(
      const FusionHelperBase* graph_group_fusion_helper,
      const OpGroupPtr& group,
      const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)>& EnableFuse)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        group_(group),
        EnableFuse_(EnableFuse),
        fuse_helper_(new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  GraphGroupLightwareFusePassCtx(const FusionHelperBase* graph_group_fusion_helper,
                                 const OpGroupPtr& group,
                                 const std::function<void(const OpGroupList& candidates)>& EnableFuseList)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        group_(group),
        EnableFuseList_(EnableFuseList),
        fuse_helper_(new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  const OpGroupPtr& PickOpGroup() const override { return group_; }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void EnableFuse(const OpGroupPtr& first, const OpGroupPtr& second) override { EnableFuse_(first, second); }

  void EnableFuse(const OpGroupList& candidates) override { EnableFuseList_(candidates); }

  const FusionHelperBase& graph_group_fusion_helper() const { return *graph_group_fusion_helper_; }

 private:
  const FusionHelperBase* graph_group_fusion_helper_;
  const OpGroupPtr group_;
  const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)> EnableFuse_;
  const std::function<void(const OpGroupList& candidates)> EnableFuseList_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

class InputFusePassCtx : public FusePassCtx {
 public:
  virtual ~InputFusePassCtx() {}

  virtual const std::unordered_set<GroupPtr>& PickConsumersWithSameInputs() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void EnableFuse(const OpGroupPtr& first, const OpGroupPtr& second) = 0;

  virtual void EnableFuse(const OpGroupList& candidates) = 0;

 protected:
  InputFusePassCtx() = default;
};

class GraphGroupInputFusePassCtx final : public InputFusePassCtx {
 public:
  GraphGroupInputFusePassCtx(const FusionHelperBase* graph_group_fusion_helper,
                             const std::unordered_set<GroupPtr>& groups,
                             const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)>& EnableFuse)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        groups_(groups),
        EnableFuse_(EnableFuse),
        fuse_helper_(new GraphGroupFuseHelper<GraphGroupInputFusePassCtx>(this)) {}

  GraphGroupInputFusePassCtx(const FusionHelperBase* graph_group_fusion_helper,
                             const std::unordered_set<GroupPtr>& groups,
                             const std::function<void(const OpGroupList& candidates)>& EnableFuseList)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        groups_(groups),
        EnableFuseList_(EnableFuseList),
        fuse_helper_(new GraphGroupFuseHelper<GraphGroupInputFusePassCtx>(this)) {}

  const std::unordered_set<GroupPtr>& PickConsumersWithSameInputs() const override { return groups_; }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void EnableFuse(const OpGroupPtr& first, const OpGroupPtr& second) override { EnableFuse_(first, second); }

  void EnableFuse(const OpGroupList& candidates) override { EnableFuseList_(candidates); }

  const FusionHelperBase& graph_group_fusion_helper() const { return *graph_group_fusion_helper_; }

 private:
  const FusionHelperBase* graph_group_fusion_helper_;
  const std::unordered_set<GroupPtr>& groups_;
  const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)> EnableFuse_;
  const std::function<void(const OpGroupList& candidates)> EnableFuseList_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::AllOutputsSameSize(const OpGroupPtr& first, const OpGroupPtr& second) const {
  return is_same_size(&ctx_->graph_group_fusion_helper(),
                      std::dynamic_pointer_cast<Graph::Group>(first),
                      std::dynamic_pointer_cast<Graph::Group>(second));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalElementwiseFuseReduce(const OpGroupPtr& src,
                                                                         const OpGroupPtr& dst) const {
  return honrizontal_elementwise_fuse_reduce(&ctx_->graph_group_fusion_helper(),
                                             std::dynamic_pointer_cast<Graph::Group>(src),
                                             std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_broadcast(&ctx_->graph_group_fusion_helper(),
                                    std::dynamic_pointer_cast<Graph::Group>(src),
                                    std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::HorizontalWithInjective(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return horizontal_with_injective(&ctx_->graph_group_fusion_helper(),
                                   std::dynamic_pointer_cast<Graph::Group>(src),
                                   std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ElementwiseFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return elementwise_fuse_reduce(&ctx_->graph_group_fusion_helper(),
                                 std::dynamic_pointer_cast<Graph::Group>(src),
                                 std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::BroadcastFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return broadcast_fuse_reduce(&ctx_->graph_group_fusion_helper(),
                               std::dynamic_pointer_cast<Graph::Group>(src),
                               std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::InjectiveHorizontalWithReduce(const OpGroupPtr& src,
                                                                       const OpGroupPtr& dst) const {
  return injective_horizontal_with_reduce(&ctx_->graph_group_fusion_helper(),
                                          std::dynamic_pointer_cast<Graph::Group>(src),
                                          std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseElementwise(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_elementwise(&ctx_->graph_group_fusion_helper(),
                                 std::dynamic_pointer_cast<Graph::Group>(src),
                                 std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseBroadcast(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_broadcast(&ctx_->graph_group_fusion_helper(),
                               std::dynamic_pointer_cast<Graph::Group>(src),
                               std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
bool GraphGroupFuseHelper<FusePassCtxT>::ReduceFuseReduce(const OpGroupPtr& src, const OpGroupPtr& dst) const {
  return reduce_fuse_reduce(&ctx_->graph_group_fusion_helper(),
                            std::dynamic_pointer_cast<Graph::Group>(src),
                            std::dynamic_pointer_cast<Graph::Group>(dst));
}

template <typename FusePassCtxT>
struct HorizontalFuseUtil {
  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;

  static bool DetectFusabilityByKind(FusePassCtxT* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    const KindKeyT kind_pair(src->kind(), dst->kind());
    const auto& map  = GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }

  typedef bool (*ConditionT)(FusePassCtxT* ctx, const OpGroupPtr& src, const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{OpPatternKind::kElementWise, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kReduction}, &HorizontalElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kReduction}, &IsSameSize},

        {{OpPatternKind::kInjective, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kReduction}, &IsSameSize},

        {{OpPatternKind::kReduction, framework::kElementWise}, &HorizontalElementwiseFuseReduce},
        {{OpPatternKind::kReduction, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kReduction, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kReduction, framework::kReduction}, &ReduceFuseReduce},
    };
  }

  static bool IsSameSize(FusePassCtxT* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().AllOutputsSameSize(src, dst);
  }

  static bool HorizontalElementwiseFuseReduce(FusePassCtxT* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().HorizontalElementwiseFuseReduce(src, dst);
  }

  static bool ReduceFuseReduce(FusePassCtxT* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseReduce(src, dst);
  }
};

class FusePass {
 public:
  virtual ~FusePass() = default;

  virtual const std::string FuseMode() const = 0;

  virtual int Benefit() const = 0;

 protected:
  FusePass() = default;
};

class InputFusePass : public FusePass {
 public:
  virtual ~InputFusePass() = default;

  virtual void operator()(InputFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const override final { return "InputFuse"; }

  virtual int Benefit() const = 0;

 protected:
  InputFusePass() = default;
};

class DefaultInputFusePass final : public InputFusePass {
 public:
  DefaultInputFusePass() : InputFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(InputFusePassCtx* ctx) const override {
    const auto& consumer_set = ctx->PickConsumersWithSameInputs();

    const std::unordered_set<OpGroupPtr> consumer_candidates = [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& consumer : consumer_set) {
        if (consumer->kind() == framework::kElementWise || consumer->kind() == framework::kBroadcast ||
            consumer->kind() == framework::kInjective || consumer->kind() == framework::kReduction) {
          consumers.insert(consumer);
        }
      }
      return consumers;
    }();
    if (consumer_candidates.size() <= 1) {
      return;
    }

    std::vector<OpGroupList> fusionable_consumers;
    for (auto& candidate : consumer_candidates) {
      if (ctx->fuse_helper().IsConsumerSetsReachable(candidate, consumer_candidates)) {
        continue;
      }
      if (fusionable_consumers.empty()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }
      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!HorizontalFuseUtil<InputFusePassCtx>::DetectFusabilityByKind(ctx, candidate, last)) {
          continue;
        }
        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    for (const auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        ctx->EnableFuse(groups);
      }
    }
  }
};

class LightwareFusePass : public FusePass {
 public:
  virtual ~LightwareFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const = 0;

  virtual int Benefit() const = 0;

 protected:
  LightwareFusePass() = default;
};

class HorizontalFusePass : public LightwareFusePass {
 public:
  virtual ~HorizontalFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const override final { return "HorizontalFuse"; }

  virtual int Benefit() const = 0;

 protected:
  HorizontalFusePass() = default;
};

class DefaultHorizontalFusePass final : public HorizontalFusePass {
 public:
  DefaultHorizontalFusePass() : HorizontalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer                                     = ctx->PickOpGroup();
    const std::unordered_set<OpGroupPtr> consumer_candidates = [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& pair : producer->consumer2outputs()) {
        if (pair.first->kind() == framework::kElementWise || pair.first->kind() == framework::kBroadcast ||
            pair.first->kind() == framework::kInjective || pair.first->kind() == framework::kReduction) {
          consumers.insert(pair.first);
        }
      }
      return consumers;
    }();
    if (consumer_candidates.size() <= 1) {
      return;
    }

    std::vector<OpGroupList> fusionable_consumers;
    for (auto& candidate : consumer_candidates) {
      if (ctx->fuse_helper().IsConsumerSetsReachable(candidate, consumer_candidates)) {
        continue;
      }
      if (fusionable_consumers.empty()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }
      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!HorizontalFuseUtil<LightwareFusePassCtx>::DetectFusabilityByKind(ctx, candidate, last)) {
          continue;
        }
        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    for (const auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        // Trick for BERT, maybe not required, wait for substitution from unordered_set to set
        if (groups.size() == 2) {
          OpGroupList fuse_group;
          if (std::dynamic_pointer_cast<Graph::Group>(groups[1])->group_id.substr(0, 4) == "cast" &&
              std::dynamic_pointer_cast<Graph::Group>(groups[0])->group_id == "reshape_split") {
            fuse_group.push_back(groups[1]);
            fuse_group.push_back(groups[0]);
            ctx->EnableFuse(fuse_group);
            continue;
          }
        }
        ctx->EnableFuse(groups);
      }
    }
  }
};

class VerticalFusePass : public LightwareFusePass {
 public:
  virtual ~VerticalFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const override final { return "VerticalFuse"; }

  virtual int Benefit() const = 0;

 protected:
  VerticalFusePass() = default;
};

class DefaultVerticalFusePass final : public VerticalFusePass {
 public:
  DefaultVerticalFusePass() : VerticalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer        = ctx->PickOpGroup();
    const OpGroupList consumers = [&]() {
      OpGroupList consumers;
      for (const auto& pair : producer->consumer2outputs()) {
        consumers.push_back(pair.first);
      }
      return consumers;
    }();
    if (consumers.size() == 0) {
      return;
    }

    std::vector<OpGroupPtr> candidates;
    for (int i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        break;
      }
      candidates.push_back(consumer);
    }
    if (candidates.size() == consumers.size() && producer->kind() == framework::kElementWise) {
      return;
    }

    for (int i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        VLOG(4) << "Can't fuse because detect cycle";
        continue;
      }
      ctx->EnableFuse(producer, consumer);
    }
  }

  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;
  bool DetectFusabilityByKind(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) const {
    const KindKeyT kind_pair(src->kind(), dst->kind());
    const auto& map  = GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }

  typedef bool (*ConditionT)(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{OpPatternKind::kElementWise, framework::kElementWise}, &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kElementWise, framework::kBroadcast}, &DefaultVerticalFusePass::ElementwiseFuseBroadcast},
        {{OpPatternKind::kElementWise, framework::kInjective}, &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kElementWise, framework::kReduction}, &DefaultVerticalFusePass::ElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, framework::kElementWise}, &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kBroadcast}, &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kInjective}, &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kBroadcast, framework::kReduction}, &DefaultVerticalFusePass::BroadcastFuseReduce},

        {{OpPatternKind::kInjective, framework::kElementWise}, &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, framework::kBroadcast}, &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, framework::kInjective}, &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kInjective, framework::kReduction}, &DefaultVerticalFusePass::InjectiveHorizontalWithReduce},

        {{OpPatternKind::kReduction, framework::kElementWise}, &DefaultVerticalFusePass::ReduceFuseElementwise},
        {{OpPatternKind::kReduction, framework::kBroadcast}, &DefaultVerticalFusePass::ReduceFuseBroadcast},
        {{OpPatternKind::kReduction, framework::kInjective}, &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kReduction, framework::kReduction}, &DefaultVerticalFusePass::ReduceFuseReduce},
    };
  }

  static bool IsSameSize(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().AllOutputsSameSize(src, dst);
  }

  static bool ElementwiseFuseBroadcast(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseBroadcast(src, dst);
  }

  static bool HorizontalWithInjective(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().HorizontalWithInjective(src, dst);
  }

  static bool ElementwiseFuseReduce(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseReduce(src, dst);
  }

  static bool BroadcastFuseReduce(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().BroadcastFuseReduce(src, dst);
  }

  static bool InjectiveHorizontalWithReduce(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().InjectiveHorizontalWithReduce(src, dst);
  }

  static bool ReduceFuseElementwise(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseElementwise(src, dst);
  }

  static bool ReduceFuseBroadcast(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseBroadcast(src, dst);
  }

  static bool ReduceFuseReduce(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseReduce(src, dst);
  }
};

class RecomputeFusePass : public LightwareFusePass {
 public:
  virtual ~RecomputeFusePass() = default;

  virtual void operator()(LightwareFusePassCtx* ctx) const = 0;

  virtual const std::string FuseMode() const override final { return "RecomputeFuse"; }

  virtual int Benefit() const = 0;

 protected:
  RecomputeFusePass() = default;
};

class DefaultRecomputeFusePass final : public RecomputeFusePass {
 public:
  DefaultRecomputeFusePass() : RecomputeFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer        = ctx->PickOpGroup();
    const OpGroupList consumers = [&]() {
      OpGroupList consumers;
      for (const auto& pair : producer->consumer2outputs()) {
        consumers.push_back(pair.first);
      }
      return consumers;
    }();
    // Borrows unsafe_candidates and candidates concept from origin fusion_merge_pass
    std::vector<OpGroupPtr> unsafe_candidates;
    std::vector<OpGroupPtr> candidates;
    for (int i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      unsafe_candidates.push_back(consumer);
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        continue;
      }
      candidates.push_back(consumer);
    }

    if (!candidates.empty() && unsafe_candidates.size() == consumers.size() &&
        producer->kind() == framework::kElementWise) {
      for (const auto& consumer : consumers) {
        ctx->EnableFuse(producer, consumer);
      }
    }
  }

  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;
  bool DetectFusabilityByKind(LightwareFusePassCtx* ctx, const OpGroupPtr& src, const OpGroupPtr& dst) const {
    const KindKeyT kind_pair(src->kind(), dst->kind());
    const auto& map  = DefaultVerticalFusePass::GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }
};

struct LightwareFusePassComparator {
  bool operator()(const std::shared_ptr<LightwareFusePass>& lhs, const std::shared_ptr<LightwareFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

struct InputFusePassComparator {
  bool operator()(const std::shared_ptr<InputFusePass>& lhs, const std::shared_ptr<InputFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

class FusionPassMap {
 public:
  static FusionPassMap& Instance() {
    static FusionPassMap global_fusion_pass_map;
    return global_fusion_pass_map;
  }

  bool Has(const std::string& pass_name) const { return map_.find(pass_name) != map_.end(); }

  void Insert(const std::string& pass_name, const std::shared_ptr<FusePass>& pass) {
    CHECK(!Has(pass_name)) << "FusePass " << pass_name << " has already been registered.";
    map_.insert({pass_name, pass});
  }

  std::shared_ptr<FusePass> Get(const std::string& pass_name) const {
    auto it = map_.find(pass_name);
    CHECK(it != map_.end()) << "FusePass " << pass_name << " has not been registered.";
    return it->second;
  }

  // fuse_mode: HorizontalFuse, VerticalFuse, RecomputeFuse
  std::vector<std::shared_ptr<LightwareFusePass>> GetLightwareFusePassesByMode(const std::string& fuse_mode) const {
    CHECK(fuse_mode == "HorizontalFuse" || fuse_mode == "VerticalFuse" || fuse_mode == "RecomputeFuse")
        << "fuse_mode only supports HorizontalFuse, VerticalFuse and RecomputeFuse. Please check your input modes = "
        << fuse_mode;
    std::set<std::shared_ptr<LightwareFusePass>, LightwareFusePassComparator> candidate_passes;
    for (const auto iter : map_) {
      if (fuse_mode == iter.second->FuseMode()) {
        candidate_passes.insert(std::dynamic_pointer_cast<LightwareFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<LightwareFusePass>>(candidate_passes.begin(), candidate_passes.end());
  }

  std::vector<std::shared_ptr<InputFusePass>> GetInputFusePasses() const {
    std::set<std::shared_ptr<InputFusePass>, InputFusePassComparator> candidate_passes;
    for (const auto iter : map_) {
      if (iter.second->FuseMode() == "InputFuse") {
        candidate_passes.insert(std::dynamic_pointer_cast<InputFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<InputFusePass>>(candidate_passes.begin(), candidate_passes.end());
  }

 private:
  FusionPassMap() = default;
  std::unordered_map<std::string, std::shared_ptr<FusePass>> map_;

  DISABLE_COPY_AND_ASSIGN(FusionPassMap);
};

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename PassClassT>
class FusionPassRegistrar final : public Registrar {
 public:
  explicit FusionPassRegistrar(const std::string& pass_name) {
    FusionPassMap::Instance().Insert(pass_name, std::shared_ptr<PassClassT>(new PassClassT()));
  }
};

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class GeneralFusionMergePassHelper : public FusionHelperBase {
 public:
  GeneralFusionMergePassHelper(const Graph* graph) : FusionHelperBase(graph) {
    fusion_groups_ = graph->fusion_groups;
    // init fusion relation.
    InitFusionRelation();
    // init input to consumers.
    InitInputToConsumers();
    // init fusion group index.
    InitFusionGroupsAndIndex();
  }

  GroupList operator()() {
    // run fusion merge untill no update.
    DoFusionMerge();
    for (auto& group : fusion_groups_) {
      VLOG(3) << "Fusion Group -> " << group->group_id;
      for (auto& sub_group : group->fused_sub_groups) {
        VLOG(3) << "  Fused Sub-Group -> " << sub_group->group_id;
      }
      for (const auto& pair : group->producer_groups()) {
        const auto& producer = std::dynamic_pointer_cast<Graph::Group>(pair.first);
        VLOG(3) << "  Producer -> " << producer->group_id;
      }
      for (const auto& pair : group->consumer_groups()) {
        const auto& consumer = std::dynamic_pointer_cast<Graph::Group>(pair.first);
        VLOG(3) << "  Consumer -> " << consumer->group_id;
      }
    }
    return fusion_groups_;
  }

 private:
  void DoFusionMerge() {
    VLOG(3) << "DoFusionMerge...!";
    while (DoGeneralHorizontalFusion()) {
    }
    while (DoGeneralVerticalFusion()) {
    }
    while (DoGeneralRecomputeAndVerticalFusion()) {
    }
  }

  bool DoGeneralHorizontalFusion() {
    VLOG(3) << "DoGeneralHorizontalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
    }

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoVerticalFusion(bool recompute) {
    VLOG(3) << "DoVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      if (!recompute) {
        updated |= HorizontalFusion(producer, producer->CollectConsumerGroups());
      }
      updated |= VerticalFusion(producer, producer->CollectConsumerGroups(), recompute);
    }
    // fuse input consumers
    updated |= FuseInputToConsumers();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralVerticalFusion() {
    VLOG(3) << "DoGeneralVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      updated |= GeneralHorizontalFuse(producer);
      updated |= GeneralVerticalFuse(producer);
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  bool DoGeneralRecomputeAndVerticalFusion() {
    VLOG(3) << "DoGeneralRecomputeAndVerticalFusion...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(3) << "Fusion Producer idx " << idx << " Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }
      // do horizontal fusion.
      bool recompute_success = GeneralRecomputeFuse(producer);
      updated |= recompute_success;
      if (!recompute_success) {
        updated |= GeneralVerticalFuse(producer);
      }
    }

    // fuse input consumers
    updated |= GeneralInputFuse();

    if (updated) {
      UpdateFusionGroup();
    }
    return updated;
  }

  void UpdateFusionGroup() {
    VLOG(3) << "UpdateFusionGroup...";
    GroupList fusion_groups;
    std::unordered_set<GroupPtr> fusion_groups_set;
    // update fusion_groups_
    for (auto& group : fusion_groups_) {
      if (!group->belong_groups.size()) {
        fusion_groups.push_back(group);
        fusion_groups_set.insert(group);
      }
    }
    // keep group in order
    fusion_groups_.clear();
    fusion_groups_index_.clear();
    while (!fusion_groups_set.empty()) {
      bool is_ring = true;
      for (int idx = 0; idx < fusion_groups.size(); ++idx) {
        auto& group = fusion_groups[idx];
        if (!group.get()) {
          continue;
        }

        bool exist = false;
        for (const auto& pair : group->producer_groups()) {
          const auto& producer = std::dynamic_pointer_cast<Graph::Group>(pair.first);
          if (fusion_groups_set.count(producer)) {
            VLOG(4) << group->group_id << " " << producer->group_id;
            exist = true;
            break;
          }
        }

        if (!exist) {
          fusion_groups_index_[group] = fusion_groups_.size();
          fusion_groups_.push_back(group);
          fusion_groups_set.erase(group);
          group.reset();
          is_ring = false;
          continue;
        }
      }
      if (is_ring) {
        LOG(FATAL) << "Exists Ring, Please Check!";
      }
    }
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawHorizontalFusePasses() const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode("HorizontalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>& GetHorizontalFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>> fuse_passes = RawHorizontalFusePasses();
    return fuse_passes;
  }

  void EnableFusedHorizontalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer->consumer2outputs().size() <= 1) {
      return;
    }
    const auto& fuse_passes = GetHorizontalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralHorizontalFuse(const GroupPtr& producer) {
    VLOG(3) << "GeneralHorizontalFuse handling producer : " << producer->group_id;
    const auto& GetFusableConsumerGroupLists = [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& EnableFuse = [&](const OpGroupList& candidates) { tagged_lists.push_back(candidates); };
      GraphGroupLightwareFusePassCtx fuse_ctx(this, producer, EnableFuse);
      EnableFusedHorizontalGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(std::dynamic_pointer_cast<Graph::Group>(group));
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  std::vector<std::shared_ptr<InputFusePass>> RawInputFusePasses() const {
    return FusionPassMap::Instance().GetInputFusePasses();
  }

  const std::vector<std::shared_ptr<InputFusePass>>& GetInputFusePasses() const {
    thread_local static std::vector<std::shared_ptr<InputFusePass>> fuse_passes = RawInputFusePasses();
    return fuse_passes;
  }

  void EnableFusedInputGroups(InputFusePassCtx* ctx) const {
    const auto& fuse_passes = GetInputFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool CallGeneralInputFusePass(const std::unordered_set<GroupPtr>& consumers) {
    VLOG(3) << "CallGeneralInputFusePass...!";
    const auto& GetFusableConsumerGroupLists = [&]() -> std::vector<OpGroupList> {
      std::vector<OpGroupList> tagged_lists;
      const auto& EnableFuse = [&](const OpGroupList& candidates) { tagged_lists.push_back(candidates); };
      GraphGroupInputFusePassCtx fuse_ctx(this, consumers, EnableFuse);
      EnableFusedInputGroups(&fuse_ctx);
      return tagged_lists;
    };
    const auto& GetFusableConsumerGroupList = [&]() -> std::vector<GroupList> {
      const auto& group_lists = GetFusableConsumerGroupLists();
      if (group_lists.empty()) {
        return std::vector<GroupList>{};
      }
      std::vector<GroupList> ret;
      for (const auto& group_list : group_lists) {
        GroupList tmp;
        for (const auto& group : group_list) {
          tmp.push_back(std::dynamic_pointer_cast<Graph::Group>(group));
        }
        ret.push_back(tmp);
      }
      return ret;
    };

    const auto& group_lists = GetFusableConsumerGroupList();
    if (group_lists.empty()) {
      return false;
    }
    for (const auto& group_list : group_lists) {
      HorizontalFuse(group_list);
    }

    return true;
  }

  bool HorizontalFusion(GroupPtr producer, const std::unordered_set<GroupPtr>& consumers) {
    VLOG(3) << "HorizontalFusion...!";
    if (consumers.size() <= 1) {
      return false;
    }

    std::unordered_set<GroupPtr> candidates;
    for (const auto& consumer : consumers) {
      // relation
      auto& relation = fusion_relation_map_[consumer->op_pattern_kind];
      // check horizontal relation exist
      if (!relation.horizontal_relation.size()) {
        continue;
      }
      candidates.insert(consumer);
    }

    std::vector<GroupList> fusionable_consumers;
    for (auto& candidate : candidates) {
      // check dependency
      if (IsDependencySimplify(producer, candidate, candidates)) {
        VLOG(4) << "IsDependencySimplify, Can't fuse " << candidate->group_id << ", As it depency others!";
        continue;
      }

      if (IsDependency(producer, candidate, candidates)) {
        VLOG(4) << "IsDependency, Can't fuse " << candidate->group_id << ", As it depency others!";
        continue;
      }

      if (!fusionable_consumers.size()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }

      // check each fusionable groups
      bool fusionable = false;
      auto& relation  = fusion_relation_map_[candidate->op_pattern_kind];
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!relation.horizontal_relation.count(last->op_pattern_kind)) {
          continue;
        }

        if (!relation.horizontal_relation[last->op_pattern_kind](this, candidate, last)) {
          continue;
        }

        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    bool updated = false;
    for (auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        updated = true;
        HorizontalFuse(groups);
      }
    }

    return updated;
  }

  void HorizontalFuse(const GroupList& consumers) {
    VLOG(3) << "HorizontalFuse Groups...";
    // create fusion group
    auto fused_group = std::make_shared<Graph::Group>();
    // As recompute exist which may case sub-group used by more than one time.
    std::vector<GroupPtr> repeat_sub_groups;
    std::unordered_set<GroupPtr> sub_group_set;
    // find the first consumer.
    GroupPtr first_consumer(nullptr);
    // fuse all group into fusion group.
    for (const auto& consumer : consumers) {
      VLOG(3) << "fuse consumer " << consumer->group_id << " into fused_group!";
      // update depth
      fused_group->max_depth = std::max(fused_group->max_depth, consumer->max_depth);
      fused_group->min_depth = std::min(fused_group->min_depth, consumer->min_depth);
      // update group id
      if (fused_group->group_id.size()) {
        fused_group->group_id += "_" + consumer->group_id;
      } else {
        fused_group->group_id = consumer->group_id;
      }
      // set op pattern kind
      fused_group->op_pattern_kind =
          static_cast<int>(fused_group->op_pattern_kind) >= static_cast<int>(consumer->op_pattern_kind)
              ? fused_group->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      for (auto& node : consumer->input_nodes) {
        if (fused_group->input_nodes.count(node.first)) {
          fused_group->input_nodes[node.first] += node.second;
        } else {
          fused_group->input_nodes.insert(node);
        }
      }
      // output node
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }
      // internal node
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // master node
      for (auto& node : consumer->master_nodes) {
        if (GetOpKind(node) == framework::kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }
      // insert sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          // check sub group is repeat.
          if (sub_group_set.count(sub_group)) {
            VLOG(3) << sub_group->group_id << " is repeated!";
            repeat_sub_groups.push_back(sub_group);
            continue;
          }
          // record sub group
          sub_group_set.insert(sub_group);

          // insert to fused sub group.
          fused_group->fused_sub_groups.push_back(sub_group);
          // update belongs group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      // producer group
      for (const auto& producer_and_list : consumer->producer_groups()) {
        GroupPtr producer = std::dynamic_pointer_cast<Graph::Group>(producer_and_list.first);
        (*fused_group->mut_producer_groups())[producer] += producer_and_list.second;
        // update producer's consumer
        producer->mut_consumer_groups()->erase(consumer);
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        (*producer->mut_consumer_groups())[fused_group] += {};
      }
      // consumer group
      for (const auto& gconsumer_and_list : consumer->consumer_groups()) {
        GroupPtr gconsumer = std::dynamic_pointer_cast<Graph::Group>(gconsumer_and_list.first);
        (*fused_group->mut_consumer_groups())[gconsumer] += gconsumer_and_list.second;
        // update consumer's producer
        gconsumer->mut_producer_groups()->erase(consumer);
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        (*gconsumer->mut_producer_groups())[fused_group] += {};
      }
      // belongs group
      consumer->belong_groups.insert(fused_group);

      // find the first consumer.
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id << " index in fusion_groups_index_!";
      if (first_consumer.get()) {
        if (fusion_groups_index_[consumer] < fusion_groups_index_[first_consumer]) {
          first_consumer = consumer;
        }
      } else {
        first_consumer = consumer;
      }
    }

    // if node is output nodes of sub_group, check it can't be internal node.
    for (auto& sub_group : repeat_sub_groups) {
      // check each output node in sub_group.
      for (auto& node : sub_group->output_nodes) {
        // if node is not output node of fused_group.
        if (!fused_group->output_nodes.count(node)) {
          fused_group->internal_nodes.insert(node);
        }
      }
    }

    if (static_cast<int>(framework::kReduction) > static_cast<int>((consumers.back())->op_pattern_kind)) {
      auto consumer = consumers.back();

      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }
    } else {
      for (auto consumer = consumers.rbegin(); consumer != consumers.rend(); ++consumer) {
        Node* master_node = nullptr;
        for (auto& node : (*consumer)->master_nodes) {
          if (GetOpKind(node) != framework::kReduction) {
            master_node = node;
            break;
          }
        }
        if (master_node) {
          VLOG(3) << "Insert Master node : " << master_node->id() << " into group : " << fused_group->group_id;
          fused_group->master_nodes.insert(master_node);
          break;
        }
      }
    }

    auto postion                      = fusion_groups_index_[first_consumer];
    fusion_groups_[postion]           = fused_group;
    fusion_groups_index_[fused_group] = postion;

    CHECK(fused_group->output_nodes.size()) << "No output node is found, " << fused_group->group_id;
  }

  bool VerticalFusion(GroupPtr& producer, const std::unordered_set<GroupPtr>& consumers, bool recompute) {
    VLOG(3) << "VerticalFusion, Number of Consumers : " << consumers.size();
    auto& relation = fusion_relation_map_[producer->op_pattern_kind];
    // if producer can't fuse others
    if (!relation.vertical_relation.size()) {
      return false;
    }

    std::unordered_set<GroupPtr> fuse_consumers_unsafe;
    std::unordered_set<GroupPtr> fuse_consumers;
    for (const auto& consumer : consumers) {
      VLOG(4) << "Check consuemr " << consumer->group_id << " can fuse to producer " << producer->group_id;
      // if can't fuse
      if (!relation.vertical_relation.count(consumer->op_pattern_kind)) {
        VLOG(4) << "Can't fuse producer " << producer->group_id << " consumer " << consumer->group_id;
        continue;
      }

      // if condition function is false
      if (!relation.vertical_relation[consumer->op_pattern_kind](this, producer, consumer)) {
        VLOG(4) << "Can't fuse producer " << producer->group_id << " consumer " << consumer->group_id;
        continue;
      }

      fuse_consumers_unsafe.insert(consumer);

      if (IsDependencySimplify(producer, consumer, consumers)) {
        VLOG(4) << "IsDependencySimplify, Consumer " << consumer->group_id << " can't be master fused group!";
        continue;
      }

      if (IsDependency(producer, consumer, consumers)) {
        VLOG(4) << "IsDependency, Consumer " << consumer->group_id << " can't be master fused group!";
        continue;
      }

      fuse_consumers.insert(consumer);
    }

    VLOG(3) << "VerticalFusion, Number of fuse Consumers : " << fuse_consumers.size();
    VLOG(3) << "VerticalFusion, Number of unsafe fuse Consumers : " << fuse_consumers.size();

    if (fuse_consumers.size() == 0) {
      return false;
    }
    // if can_fuse_consumers == consumers
    // if producer op kind == kElementwise
    // if use recompute
    if (fuse_consumers_unsafe.size() == producer->consumer_groups().size() &&
        producer->op_pattern_kind == framework::kElementWise) {
      if (!recompute) {
        return false;
      } else {
        RecomputeEleGraph(producer, fuse_consumers_unsafe);
        VerticalFuse(producer, fuse_consumers_unsafe);
        return true;
      }
    }

    if (fuse_consumers.size()) {
      SelectConsumerToFuse(producer, fuse_consumers);
    }

    // if fusionable consumers exist
    if (fuse_consumers.size()) {
      VerticalFuse(producer, fuse_consumers);
      return true;
    }

    return false;
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawVerticalFusePasses() const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode("VerticalFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>& GetVerticalFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>> fuse_passes = RawVerticalFusePasses();
    return fuse_passes;
  }

  void TagVerticalGroups(LightwareFusePassCtx* ctx) const {
    const auto& producer = ctx->PickOpGroup();
    if (producer->consumer2outputs().empty()) {
      return;
    }
    const auto& fuse_passes = GetVerticalFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralVerticalFuse(GroupPtr& producer) {
    VLOG(3) << "GeneralVerticalFuse handling producer : " << producer->group_id;
    using GroupSets                           = std::set<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& EnableFuse = [&](const OpGroupPtr& first, const OpGroupPtr& second) {
        tagged_sets.insert(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(this, producer, EnableFuse);
      TagVerticalGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet = [&]() -> std::unordered_set<GroupPtr> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(std::dynamic_pointer_cast<Graph::Group>(group_pair.second));
      }
      return ret;
    };

    bool update          = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size()) {
      SelectConsumerToFuse(producer, consumer_groups);
    }
    if (consumer_groups.size() > 0) {
      VerticalFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void VerticalFuse(GroupPtr& producer, std::unordered_set<GroupPtr>& fusionable_consumers) {
    VLOG(3) << "VerticalFuse...!";
    GroupList fused_groups;
    GroupPtr master_fuesd_group(nullptr);
    for (auto& consumer : fusionable_consumers) {
      auto fused_group = std::make_shared<Graph::Group>();
      // update depth using consumer depth.
      fused_group->max_depth = std::max(producer->max_depth, consumer->max_depth);
      fused_group->min_depth = std::min(producer->min_depth, consumer->min_depth);
      // update group id
      fused_group->group_id = producer->group_id + "_" + consumer->group_id;
      VLOG(3) << "fuse producer " << producer->group_id << " into consumer " << consumer->group_id;
      // fuse producer into fusion group
      fused_group->op_pattern_kind =
          static_cast<int>(producer->op_pattern_kind) >= static_cast<int>(consumer->op_pattern_kind)
              ? producer->op_pattern_kind
              : consumer->op_pattern_kind;
      // input nodes
      fused_group->input_nodes = producer->input_nodes;

      // internal nodes
      if (producer->fused_sub_groups.size()) {
        for (auto& node : producer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }
      // convert producer's output node to internal.
      for (auto node : producer->output_nodes) {
        // if node is used more than 1 time.
        if (consumer->input_nodes.count(node)) {
          if (consumer->input_nodes[node] > 1 && node->inlinks().size() > 0) {
            fused_group->internal_nodes.insert(node);
          }
        }
      }
      // master nodes
      for (auto& node : producer->master_nodes) {
        if (GetOpKind(node) == framework::kReduction) {
          fused_group->master_nodes.insert(node);
        }
      }

      // producer groups
      for (const auto& group_and_list : producer->producer_groups()) {
        (*fused_group->mut_producer_groups())[group_and_list.first] += group_and_list.second;
        const auto& group = std::dynamic_pointer_cast<Graph::Group>(group_and_list.first);
        // update producer's producer's consumer
        group->mut_consumer_groups()->erase(producer);
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        (*group->mut_consumer_groups())[fused_group] += {};
      }

      // sub groups
      if (producer->fused_sub_groups.size()) {
        for (auto& group : producer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(group);
          // update belong group
          group->belong_groups.erase(producer);
          group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(producer);
      }
      producer->belong_groups.insert(fused_group);

      // input nodes
      for (auto& input_node : consumer->input_nodes) {
        // if input node not in producer output.
        if (!producer->output_nodes.count(input_node.first)) {
          if (fused_group->input_nodes.count(input_node.first)) {
            fused_group->input_nodes[input_node.first] += input_node.second;
          } else {
            fused_group->input_nodes.insert(input_node);
          }
        }
      }

      // output nodes
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }

      // internal nodes
      if (consumer->fused_sub_groups.size()) {
        for (auto& node : consumer->internal_nodes) {
          fused_group->internal_nodes.insert(node);
        }
      }

      // master nodes
      for (auto& node : consumer->master_nodes) {
        fused_group->master_nodes.insert(node);
      }

      // producer nodes
      for (const auto& group_and_list : consumer->producer_groups()) {
        if (group_and_list.first.get() != producer.get()) {
          (*fused_group->mut_producer_groups())[group_and_list.first] += group_and_list.second;
          const GroupPtr& group = std::dynamic_pointer_cast<Graph::Group>(group_and_list.first);
          // update consumer's producer's consumer
          group->mut_consumer_groups()->erase(consumer);
          // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
          (*group->mut_consumer_groups())[fused_group] += {};
        }
      }

      // consumer nodes
      for (const auto& group_and_list : consumer->consumer_groups()) {
        (*fused_group->mut_consumer_groups())[group_and_list.first] += group_and_list.second;
        const GroupPtr& group = std::dynamic_pointer_cast<Graph::Group>(group_and_list.first);
        // update consumer's consumer's producer
        group->mut_producer_groups()->erase(consumer);
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        (*group->mut_producer_groups())[fused_group] += {};
      }

      // sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          if (std::find(fused_group->fused_sub_groups.begin(), fused_group->fused_sub_groups.end(), sub_group) ==
              fused_group->fused_sub_groups.end()) {
            fused_group->fused_sub_groups.push_back(sub_group);
          }
          // update belong group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      consumer->belong_groups.insert(fused_group);

      fused_groups.push_back(fused_group);
      CHECK(fusion_groups_index_.count(consumer))
          << "Can't find consumer " << consumer->group_id << " index in fusion_groups_index_!";
      auto postion                      = fusion_groups_index_[consumer];
      fusion_groups_[postion]           = fused_group;
      fusion_groups_index_[fused_group] = postion;

      if (!master_fuesd_group.get()) {
        master_fuesd_group = fused_group;
      }
      CHECK(fused_group->output_nodes.size()) << "No output node is found, " << fused_group->group_id;
    }

    for (auto& node : producer->output_nodes) {
      bool be_output = true;
      for (const auto& consumer_and_list : producer->consumer_groups()) {
        const auto& consumer = std::dynamic_pointer_cast<Graph::Group>(consumer_and_list.first);
        // if consumer is in fusionable.
        if (fusionable_consumers.count(consumer)) {
          if (consumer->input_nodes.count(node)) {
            be_output = false;
          }
          continue;
        }
        // if consumer is not in fusionable.
        if (consumer->input_nodes.count(node)) {
          be_output = true;
          break;
        }
        // others node is as graph output.
      }

      if (output_nodes_set_.count(node)) {
        be_output = true;
      }

      if (be_output) {
        VLOG(4) << "Insert Id " << node->id() << " Into Group " << master_fuesd_group->group_id;
        master_fuesd_group->output_nodes.insert(node);
      }
    }
    // insert unfusionable consumer groups
    for (const auto& consumer_and_list : producer->consumer_groups()) {
      const auto& consumer = std::dynamic_pointer_cast<Graph::Group>(consumer_and_list.first);
      if (fusionable_consumers.count(consumer)) {
        continue;
      }
      (*master_fuesd_group->mut_consumer_groups())[consumer_and_list.first] += consumer_and_list.second;
      // update consumer's producer
      consumer->mut_producer_groups()->erase(producer);
      // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
      (*consumer->mut_producer_groups())[master_fuesd_group] += {};
    }
  }

  std::vector<std::shared_ptr<LightwareFusePass>> RawRecomputeFusePasses() const {
    return FusionPassMap::Instance().GetLightwareFusePassesByMode("RecomputeFuse");
  }

  const std::vector<std::shared_ptr<LightwareFusePass>>& GetRecomputeFusePasses() const {
    thread_local static std::vector<std::shared_ptr<LightwareFusePass>> fuse_passes = RawRecomputeFusePasses();
    return fuse_passes;
  }

  void TagRecomputeGroups(LightwareFusePassCtx* ctx) const {
    const auto& fuse_passes = GetRecomputeFusePasses();
    for (const auto& fuse_pass : fuse_passes) {
      (*fuse_pass)(ctx);
    }
  }

  bool GeneralRecomputeFuse(GroupPtr& producer) {
    VLOG(3) << "GeneralRecomputeFuse handling producer : " << producer->group_id;
    using GroupSets                           = std::set<std::pair<OpGroupPtr, OpGroupPtr>>;
    const auto& GetFusableConsumerOpGroupSets = [&]() -> GroupSets {
      GroupSets tagged_sets;
      const auto& EnableFuse = [&](const OpGroupPtr& first, const OpGroupPtr& second) {
        tagged_sets.insert(std::make_pair(first, second));
      };
      GraphGroupLightwareFusePassCtx fuse_ctx(this, producer, EnableFuse);
      TagRecomputeGroups(&fuse_ctx);
      return tagged_sets;
    };

    auto GetFusableConsumerGroupSet = [&]() -> std::unordered_set<GroupPtr> {
      const auto& group_sets = GetFusableConsumerOpGroupSets();
      if (group_sets.empty()) {
        return {};
      }
      std::unordered_set<GroupPtr> ret;
      for (const auto& group_pair : group_sets) {
        ret.insert(std::dynamic_pointer_cast<Graph::Group>(group_pair.second));
      }
      return ret;
    };

    bool update          = false;
    auto consumer_groups = GetFusableConsumerGroupSet();
    if (consumer_groups.size() > 0) {
      CHECK(consumer_groups.size() == producer->mut_consumer_groups()->size())
          << "Recompute requires fuse all consumers!";
      RecomputeFuse(producer, consumer_groups);
      update = true;
    }
    return update;
  }

  void RecomputeFuse(GroupPtr& producer, std::unordered_set<GroupPtr>& fusionable_consumers) {
    VerticalFuse(producer, fusionable_consumers);
  }

  void RecomputeEleGraph(const GroupPtr& producer, std::unordered_set<GroupPtr>& fusionable_consumers) {
    if (producer->op_pattern_kind != framework::kElementWise) {
      SelectConsumerToFuse(producer, fusionable_consumers);
    }
  }

  void SelectConsumerToFuse(const GroupPtr& producer, std::unordered_set<GroupPtr>& fusionable_consumers) {
    // if is const op
    if (is_const_group(this, producer)) {
      std::unordered_set<GroupPtr> candidates;
      for (auto& consumer : fusionable_consumers) {
        // if can be output node.
        if (is_same_shape(this, producer, consumer)) {
          candidates.insert(consumer);
        } else {
          VLOG(4) << "Fuse Producer : " << producer->group_id << " into Consumer : " << consumer->group_id;
          consumer->group_id = producer->group_id + "_" + consumer->group_id;
          // just merge the node into group.
          auto& sub_group     = consumer->fused_sub_groups.front();
          sub_group->group_id = producer->group_id + "_" + sub_group->group_id;
          sub_group->nodes.insert(sub_group->nodes.begin(), producer->CollectNodes()[0]);
          sub_group->nodes_set.insert(producer->CollectNodes()[0]);
          // remove depency.
          consumer->input_nodes.erase(producer->CollectNodes()[0]);
          consumer->mut_producer_groups()->erase(producer);
          producer->mut_consumer_groups()->erase(consumer);
        }
      }

      CHECK_GE(producer->consumer_groups().size(), candidates.size());
      if (producer->consumer_groups().size() == 0 && candidates.size() == 0 &&
          output_nodes_set_.count(producer->CollectNodes()[0]) == 0) {
        producer->belong_groups.insert(*fusionable_consumers.begin());
      }

      fusionable_consumers = candidates;
      return;
    }
    // 1 to 1 fusion.
    if (producer->consumer_groups().size() == 1) {
      return;
    }

    if (FLAGS_enhance_vertical_fusion_with_recompute) {
      std::vector<GroupPtr> candidates;
      for (auto& consumer : fusionable_consumers) {
        if (consumer->op_pattern_kind == framework::kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto producer_output_shape       = this->GetNodeDataShape(*producer->output_nodes.begin());
        auto consumer_output_shape       = this->GetNodeDataShape(*consumer->output_nodes.begin());
        auto consumer_master_input_shape = this->GetNodeInputShape(*(consumer->master_nodes.begin()));
        int producer_output_numel =
            std::accumulate(producer_output_shape.begin(), producer_output_shape.end(), 1, std::multiplies<int>());
        int consumer_output_numel =
            std::accumulate(consumer_output_shape.begin(), consumer_output_shape.end(), 1, std::multiplies<int>());
        int consumer_master_input_numel = std::accumulate(
            consumer_master_input_shape.begin(), consumer_master_input_shape.end(), 1, std::multiplies<int>());
        if (producer_output_numel == consumer_output_numel) {
          candidates.push_back(consumer);
          continue;
        }

        if (producer->op_pattern_kind != framework::kInjective && consumer->op_pattern_kind == framework::kReduction &&
            producer_output_numel == consumer_master_input_numel) {
          candidates.push_back(consumer);
        }
      }
      sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->op_pattern_kind < rhs->op_pattern_kind;
      });

      fusionable_consumers.clear();
      if (candidates.size()) {
        fusionable_consumers.insert(*candidates.begin());
      }
    } else {
      std::vector<GroupPtr> candidates;
      for (auto& consumer : fusionable_consumers) {
        if (consumer->op_pattern_kind == framework::kElementWise) {
          candidates.push_back(consumer);
          continue;
        }

        auto shape0 = this->GetNodeDataShape(*producer->output_nodes.begin());
        auto shape1 = this->GetNodeDataShape(*consumer->output_nodes.begin());

        if (std::accumulate(shape0.begin(), shape0.end(), 1, std::multiplies<int>()) ==
            std::accumulate(shape1.begin(), shape1.end(), 1, std::multiplies<int>())) {
          candidates.push_back(consumer);
        }
      }

      fusionable_consumers.clear();
      if (candidates.size()) {
        fusionable_consumers.insert(candidates.front());
      }
    }
  }

  bool IsDependency(const GroupPtr& producer_g,
                    const GroupPtr& consumer,
                    const std::unordered_set<GroupPtr>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);

    std::unordered_set<GroupPtr> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (const auto& producer_and_list : candidate->producer_groups()) {
        if (producer_and_list.first.get() == producer_g.get()) {
          continue;
        }
        const auto& producer = std::dynamic_pointer_cast<Graph::Group>(producer_and_list.first);
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool IsDependencySimplify(const GroupPtr& producer_g,
                            const GroupPtr& consumer,
                            const std::unordered_set<GroupPtr>& consumers) {
    std::queue<GroupPtr> candidates;
    candidates.push(consumer);
    // check upper.
    int check_upper_depth = producer_g.get() ? producer_g->max_depth : INT_MAX;
    std::unordered_set<GroupPtr> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (auto& producer_and_list : candidate->producer_groups()) {
        if (producer_and_list.first.get() == producer_g.get()) {
          continue;
        }
        const auto& producer = std::dynamic_pointer_cast<Graph::Group>(producer_and_list.first);
        if (producer->min_depth > check_upper_depth) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

  bool FuseInputToConsumers() {
    VLOG(3) << "FuseInputToConsumers...!";
    auto updated = false;
    UpdateInputToConsumers();
    GroupPtr producer(nullptr);
    for (auto& input_consumers : input_to_consumers_) {
      // if group set size == 1.
      if (input_consumers.second.size() == 1) {
        continue;
      }
      // do horizontal fusion.
      auto st = HorizontalFusion(producer, input_consumers.second);
      if (st) {
        // fused consumers, update
        UpdateInputToConsumers();
      }
      updated |= st;
    }

    return updated;
  }

  bool GeneralInputFuse() {
    VLOG(3) << "GeneralInputFuse...!";
    auto updated = false;
    UpdateInputToConsumers();
    for (auto& input_consumers : input_to_consumers_) {
      // if group set size == 1.
      if (input_consumers.second.size() == 1) {
        continue;
      }
      // do input fusion.
      auto st = CallGeneralInputFusePass(input_consumers.second);
      if (st) {
        // fused consumers, update
        UpdateInputToConsumers();
      }
      updated |= st;
    }

    return updated;
  }

  void UpdateInputToConsumers() {
    for (auto& input_consumers : input_to_consumers_) {
      auto& consumers = input_consumers.second;
      std::unordered_set<GroupPtr> updated_consumers;
      for (auto& consumer : consumers) {
        std::queue<GroupPtr> fused_groups;
        fused_groups.push(consumer);
        while (!fused_groups.empty()) {
          auto& cur = fused_groups.front();
          fused_groups.pop();
          // if group is sub group
          if (cur->belong_groups.empty()) {
            updated_consumers.insert(cur);
          } else {
            for (auto& belong_group : cur->belong_groups) {
              if (belong_group->group_id == cur->group_id) {
                updated_consumers.insert(belong_group);
              } else {
                fused_groups.push(belong_group);
              }
            }
          }
        }
      }
      consumers = updated_consumers;
    }
  }

  void InitInputToConsumers() {
    VLOG(3) << "InitInputToConsumers...!";
    // init input data node -> fusion group map.
    for (auto& group : fusion_groups_) {
      for (auto& node : group->nodes_set) {
        // collect producer node data.
        auto producer_node_datas = GetProducerNodeData(node);
        for (auto& node_data : producer_node_datas) {
          // node data's source node is null.
          if (!node_data->source_node.get()) {
            // insert group to set.
            input_to_consumers_[node_data].insert(group);
          }
        }
      }
    }
  }

  void InitFusionGroupsAndIndex() {
    VLOG(3) << "InitFusionGroupsAndIndex...!";
    // init the postion of groups in fusion groups.
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto group        = fusion_groups_[idx];
      auto belong_group = std::make_shared<Graph::Group>();
      // copy from group.
      belong_group->max_depth                = group->depth;
      belong_group->min_depth                = group->depth;
      belong_group->group_id                 = group->group_id;
      belong_group->input_nodes              = group->input_nodes;
      belong_group->output_nodes             = group->output_nodes;
      belong_group->op_pattern_kind          = group->op_pattern_kind;
      belong_group->master_nodes             = group->master_nodes;
      (*belong_group->mut_producer_groups()) = group->producer_groups();
      (*belong_group->mut_consumer_groups()) = group->consumer_groups();
      belong_group->fused_sub_groups.push_back(group);
      group->belong_groups.insert(belong_group);
      // replace group to fused_group
      fusion_groups_[idx] = belong_group;
      // record idx
      fusion_groups_index_[belong_group] = idx;
    }

    // update producer and consumer.
    for (auto& group : fusion_groups_) {
      std::unordered_map<OpGroupPtr, TensorInterfaceList> producers;
      std::unordered_map<OpGroupPtr, TensorInterfaceList> consumers;

      for (auto& producer_and_list : group->producer_groups()) {
        const auto& producer = std::dynamic_pointer_cast<Graph::Group>(producer_and_list.first);
        CHECK(producer->belong_groups.size());
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        producers[*producer->belong_groups.begin()] += {};
      }

      for (auto& consumer_and_list : group->consumer_groups()) {
        const auto& consumer = std::dynamic_pointer_cast<Graph::Group>(consumer_and_list.first);
        CHECK(consumer->belong_groups.size());
        // TODO: Do not add any TensorInterface into any TensorInterfaceList in this file which will be deprecated.
        consumers[*consumer->belong_groups.begin()] += {};
      }
      CHECK_EQ(group->producer_groups().size(), producers.size());
      CHECK_EQ(group->consumer_groups().size(), consumers.size());
      (*group->mut_producer_groups()) = producers;
      (*group->mut_consumer_groups()) = consumers;
    }
  }

  void InitFusionRelation() {
    VLOG(3) << "InitFusionRelation...!";
    // kElementWise
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kElementWise];
      // horizontal
      relation.horizontal_relation = {{framework::kElementWise, is_same_size},
                                      // element-wise and broadcast op must be horizontal relation.
                                      {OpPatternKind::kBroadcast, is_same_size},
                                      // element-wise and injective op must be horizontal relation.
                                      {OpPatternKind::kInjective, is_same_size},
                                      // element-wise and reduce op must be horizontal relation.
                                      {OpPatternKind::kReduction, honrizontal_elementwise_fuse_reduce}};
      // vertical
      relation.vertical_relation = {{OpPatternKind::kElementWise, is_same_size},
                                    // element-wise and broadcast can be vertical/horizontal relation.
                                    {OpPatternKind::kBroadcast, elementwise_fuse_broadcast},
                                    // element-wise and injective op must be horizontal relation.
                                    {OpPatternKind::kInjective, horizontal_with_injective},
                                    // element-wise and reduce can be vertical/horizontal relation.
                                    {OpPatternKind::kReduction, elementwise_fuse_reduce}};
    }
    // kBroadcast
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kBroadcast];
      // horizontal
      relation.horizontal_relation = {// broadcast and element-wise op must be horizontal relation.
                                      {framework::kElementWise, is_same_size},
                                      // broadcast and broadcast op must be horizontal relation.
                                      {framework::kBroadcast, is_same_size},
                                      // broadcast and injective op must be horizontal relation.
                                      {OpPatternKind::kInjective, is_same_size},
                                      // broadcast and reduce op must be horizontal relation.
                                      {OpPatternKind::kReduction, is_same_size}};
      // vertical
      relation.vertical_relation = {// broadcast and element-wise op must be vertical relation.
                                    {OpPatternKind::kElementWise, is_same_size},
                                    // broadcast and broadcast op must be horizontal relation.
                                    {OpPatternKind::kBroadcast, is_same_size},
                                    // broadcast and injective op must be horizontal relation.
                                    {OpPatternKind::kInjective, horizontal_with_injective},
                                    // broadcast and reduce must be vertical relation.
                                    {OpPatternKind::kReduction, broadcast_fuse_reduce}};
    }
    // kInjective
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kInjective];
      // horizontal
      relation.horizontal_relation = {// injective and element-wise op must be horizontal relation.
                                      {OpPatternKind::kElementWise, is_same_size},
                                      // injective and broadcast op must be horizontal relation.
                                      {OpPatternKind::kBroadcast, is_same_size},
                                      // injective and injective op must be horizontal relation.
                                      {OpPatternKind::kInjective, is_same_size},
                                      // injective and reduce must be horizontal relation.
                                      {OpPatternKind::kReduction, is_same_size}};
      // vertical
      relation.vertical_relation = {// injective and element-wise op must be horizontal relation.
                                    {OpPatternKind::kElementWise, is_same_size},
                                    // injective and broadcast op must be horizontal relation.
                                    {OpPatternKind::kBroadcast, is_same_size},
                                    // injective and injective op must be horizontal relation.
                                    {OpPatternKind::kInjective, horizontal_with_injective},
                                    // injective and reduce can be horizontal/vertical relation.
                                    {OpPatternKind::kReduction, injective_horizontal_with_reduce}};
    }
    // kReduction
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kReduction];
      // horizontal
      relation.horizontal_relation = {// reduce and element-wise op must be horizontal relation.
                                      {OpPatternKind::kElementWise, honrizontal_elementwise_fuse_reduce},
                                      // reduce and broadcast op must be horizontal relation.
                                      {OpPatternKind::kBroadcast, is_same_size},
                                      // reduce and injective op must be horizontal relation.
                                      {OpPatternKind::kInjective, is_same_size},
                                      // reduce and reduce must be horizontal relation.
                                      {OpPatternKind::kReduction, reduce_fuse_reduce}};
      // vertical
      relation.vertical_relation = {// reduce and elementwise can be horizontal/vertical relation.
                                    {OpPatternKind::kElementWise, reduce_fuse_elementwise},
                                    // reduce and broadcast op must be horizontal relation.
                                    {OpPatternKind::kBroadcast, reduce_fuse_broadcast},
                                    // reduce and injective op must be horizontal relation.
                                    {OpPatternKind::kInjective, horizontal_with_injective},
                                    // reduce and reduce must be horizontal relation.
                                    {OpPatternKind::kReduction, reduce_fuse_reduce}};
    }
  }

  GroupList fusion_groups_;
  std::unordered_map<GroupPtr, int> fusion_groups_index_;
  std::unordered_map<NodeData*, std::unordered_set<GroupPtr>> input_to_consumers_;

  struct Relation {
    std::unordered_map<framework::OpPatternKind, ConditionFunction> vertical_relation;
    std::unordered_map<framework::OpPatternKind, ConditionFunction> horizontal_relation;
  };
  std::unordered_map<framework::OpPatternKind, Relation> fusion_relation_map_;
};

void GeneralFusionMergePassInternal(Graph* graph) {
  if (graph->fusion_groups.size() <= 1) {
    VLOG(3) << "Don't do Fusoin Merge Pass...!";
    return;
  }

  GeneralFusionMergePassHelper fusion_merge_pass_helper(graph);
  graph->fusion_groups = fusion_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(GeneralFusionMergePass) {
  CINN_REGISTER_PASS(GeneralFusionMergePass)
      .describe(
          "Fusion Merge Pass which performs Fusion-Ops fusion, Producer Fusion-Ops are fused into Consumer Fusion-Ops "
          "with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::GeneralFusionMergePassInternal);

  return true;
}

CINN_REGISTER_FUSION_PASS(DefaultHorizontalFusePass, cinn::hlir::pass::DefaultHorizontalFusePass);
CINN_REGISTER_FUSION_PASS(DefaultVerticalFusePass, cinn::hlir::pass::DefaultVerticalFusePass);
CINN_REGISTER_FUSION_PASS(DefaultRecomputeFusePass, cinn::hlir::pass::DefaultRecomputeFusePass);
CINN_REGISTER_FUSION_PASS(DefaultInputFusePass, cinn::hlir::pass::DefaultInputFusePass);

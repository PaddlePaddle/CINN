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

#pragma once

#include <memory>
#include <set>

#include "cinn/common/shared.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

class Node {
 public:
  Node()            = default;
  virtual ~Node()   = default;
  Node(const Node&) = delete;

  virtual bool Tell(Node* instr) const { return false; }

  int16_t id() const { return id_; }
  void set_id(int16_t id) { id_ = id; }
  void set_label(const char* label) { label_ = label; }
  const char* label() const { return label_; }

 private:
  int16_t id_{-1};
  const char* label_{};
};

std::ostream& operator<<(std::ostream& os, const Node& node);

class Digraph;

class ProgramVar final : public Node {
 public:
  ProgramVar(const Digraph& prog, const Variable& var) : prog_{&prog}, var_{var} {}
  const Variable* raw() const { return &var_; }
  Digraph const* prog() const { return prog_; }

 private:
  Variable var_;
  Digraph const* prog_{};
};

class ProgramInstr final : public Node {
 public:
  ProgramInstr(const Instruction& instr) : instr_{instr} {}
  const Instruction* raw() const { return &instr_; }

 private:
  Instruction instr_;
};

class PatternVar final : public Node {
 public:
  bool Tell(Node* var) const override;
  PatternVar* set_label(const char* label) {
    Node::set_label(label);
    return this;
  }
  PatternVar* Assert(const std::function<bool(ProgramVar*)>& teller) {
    tellers_.emplace_back(teller);
    return this;
  }
  bool external() const { return external_; }

 private:
  bool external_{true};
  std::vector<std::function<bool(ProgramVar*)>> tellers_;
};

class PatternInstr final : public Node {
 public:
  PatternInstr(const char* type) : type_{type} {
    tellers_.emplace_back([=](ProgramInstr* instr) -> bool { return instr->raw()->get()->op_type == type_; });
  }
  PatternInstr* Assert(const std::function<bool(ProgramInstr*)>& teller) {
    tellers_.emplace_back(teller);
    return this;
  }
  const char* type() const { return type_; }
  PatternInstr* set_label(const char* label) {
    Node::set_label(label);
    return this;
  }

  bool Tell(Node* instr) const override;

 private:
  const char* type_{};
  std::vector<std::function<bool(ProgramInstr*)>> tellers_;
};

class Target {
 public:
  Target(Node* end, int16_t idx) : end_{end}, var_idx_{idx} {}
  explicit Target(Node* end) : end_{end} {}
  Node* end() const { return end_; }
  int16_t var_idx() const { return var_idx_; }

 private:
  Node* end_{};
  int16_t var_idx_{-1};
};

bool operator<(const Target& lhs, const Target& rhs);

struct NodeLessThan {
  bool operator()(const Node* lhs, const Node* rhs) const;

  template <typename T, typename = std::enable_if_t<std::is_base_of<Node, T>::value>>
  bool operator()(const std::unique_ptr<T>& lhs, const std::unique_ptr<T>& rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }

  bool operator()(const std::pair<Node const*, Node const*>& lhs, const std::pair<Node const*, Node const*>& rhs) const;
};

class Adjacent;

class EdgeIterable {
 public:
  class iterator : public std::iterator<std::forward_iterator_tag, std::pair<Node*, Node*>> {
   public:
    using internal_t = std::set<std::pair<Node*, Node*>, NodeLessThan>::iterator;
    iterator(internal_t it) : cur_{it} {}
    iterator& operator++() {
      ++cur_;
      return *this;
    }
    iterator& operator--() {
      --cur_;
      return *this;
    }
    const std::pair<Node*, Node*>& operator*() { return *cur_; }
    internal_t operator->() { return cur_; }
    bool operator==(const iterator& other) const { return cur_ == other.cur_; }
    bool operator!=(const iterator& other) const { return !(*this == other); }

   private:
    internal_t cur_;
  };

  EdgeIterable() = default;
  explicit EdgeIterable(const Adjacent& adj);

  iterator begin() { return iterator{edges_.begin()}; }
  iterator end() { return iterator{edges_.end()}; }

 private:
  std::set<std::pair<Node*, Node*>, NodeLessThan> edges_;
};

class Adjacent {
 public:
  using edge_t = std::pair<Node*, Node*>;
  size_t size() const { return adj_.size(); }
  void Add(Node* start) {
    if (adj_.find(start) == adj_.end()) {
      adj_[start] = {};
    }
  }
  const std::set<Target>& targets(Node* start) const { return adj_.at(start); }
  void Add(Node* start, Node* end, int16_t idx) { adj_[start].emplace(Target(end, idx)); }
  bool HasEdge(Node* start, Node* end) const;
  EdgeIterable edges() const { return EdgeIterable(*this); }
  const std::set<Target>& GetTargets(Node* start) const { return adj_.at(start); }

 private:
  friend class EdgeIterable;
  std::map<Node*, std::set<Target>, NodeLessThan> adj_;
};

class Digraph {
 public:
  Digraph()               = default;
  Digraph(const Digraph&) = delete;
  Digraph(Digraph&&)      = default;
  Digraph& operator=(Digraph&&) = default;

  Node* AddNode(std::unique_ptr<Node>&& node) {
    auto* ret = node.get();
    nodes_.emplace(std::move(node));
    adj_.Add(ret);
    return ret;
  }
  template <typename... Args>
  void AddEdge(Args... args) {
    adj_.Add(std::forward<Args>(args)...);
  }
  const std::set<std::unique_ptr<Node>, NodeLessThan>& nodes() const { return nodes_; }
  const Adjacent& adj() const { return adj_; }

  // TODO: check for directed acyclic.
 private:
  std::set<std::unique_ptr<Node>, NodeLessThan> nodes_;
  Adjacent adj_;
};

class DepthFirstSearch {
 public:
  DepthFirstSearch(const Digraph& graph) : g_{&graph} {}
  const std::set<Node const*, NodeLessThan>& operator()(Node const* start) {
    marked_.clear();
    dfs(start);
    return marked_;
  }
  bool accessible(Node const* start, Node const* end) {
    marked_.clear();
    dfs(start);
    return marked_.count(end);
  }

 private:
  void dfs(Node const* start) {
    marked_.emplace(start);
    for (auto& t : g_->adj().targets(const_cast<Node*>(start))) {
      if (!marked_.count(t.end())) {
        dfs(t.end());
      }
    }
  }
  const Digraph* g_;
  std::set<Node const*, NodeLessThan> marked_;
};

std::ostream& operator<<(std::ostream& os, const Digraph& graph);

class GraphBuilder {
 public:
  GraphBuilder()                             = default;
  GraphBuilder(const GraphBuilder&)          = delete;
  GraphBuilder(GraphBuilder&&)               = default;
  virtual ~GraphBuilder()                    = default;
  virtual std::unique_ptr<Digraph> release() = 0;
};

class PatternBuilder final : public GraphBuilder {
 public:
  PatternVar* AddVar();

  PatternInstr* AddInstr(const char* type,
                         const std::vector<PatternVar*>& inputs,
                         const std::vector<PatternVar*>& outputs);

  int16_t cur_id() const { return cur_id_; }
  std::unique_ptr<Digraph> release() override { return std::move(graph_); }

 private:
  int16_t cur_id_{-1};
  std::unique_ptr<Digraph> graph_{std::make_unique<Digraph>()};
};

class ProgramGraphBuilder final : public GraphBuilder {
 public:
  ProgramGraphBuilder(const Program& program);
  std::unique_ptr<Digraph> release() override { return std::move(graph_); }

 private:
  void AddInstr(const Instruction& instr);
  bool VarExists(const Variable& var) const { return var_map_.count(var.get()); }
  void AddVar(const Variable& var);

  int16_t cur_id_{-1};
  std::map<const _Variable_*, ProgramVar*> var_map_;
  std::unique_ptr<Digraph> graph_{std::make_unique<Digraph>()};
};

// TODO: use a more classical algorithm.
class PatternMatcher {
 public:
  using pattern_node_t = Node;
  using program_node_t = Node;
  using pattern_map_t  = std::map<pattern_node_t const*, program_node_t const*>;
  PatternMatcher()     = default;
  void Init(const Digraph& pattern, const Digraph& program);
  std::vector<pattern_map_t> DetectPatterns();

  class HitGroup {
   public:
    const std::map<pattern_node_t const*, program_node_t const*, NodeLessThan>& roles() const { return roles_; }
    void Register(program_node_t const* node, pattern_node_t const* pat) {
      roles_[pat] = node;
      nodes_.insert(node);
    }
    bool Match(program_node_t const* node, pattern_node_t const* pat) const;
    bool operator<(const HitGroup& rhs) const { return roles_ < rhs.roles_; }
    friend std::ostream& operator<<(std::ostream& os, const HitGroup& group);

   private:
    std::map<pattern_node_t const*, program_node_t const*, NodeLessThan> roles_;
    std::set<program_node_t const*, NodeLessThan> nodes_;
  };

 private:
  void NodeMatch();
  Digraph const* program_{};
  Digraph const* pattern_{};
  std::map<pattern_node_t*, std::vector<program_node_t*>, NodeLessThan> pdnodes2nodes_;
  EdgeIterable pattern_edges_;
  std::vector<HitGroup> groups_;
};

// TODO: rewrite the program efficiently
ProgramInstr const* get_mapped_instr(const PatternMatcher::pattern_map_t& matches, const char* label);
ProgramVar const* get_mapped_var(const PatternMatcher::pattern_map_t& matches, const char* label);

}  // namespace cinn::frontend::pass

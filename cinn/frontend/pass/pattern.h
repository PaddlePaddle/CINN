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

#include <array>
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

  virtual bool Tell(const Node* instr) const { return false; }

  int16_t id() const { return id_; }
  void set_id(int16_t id) { id_ = id; }
  void set_label(const char* label) { label_ = label; }
  const char* label() const { return label_; }

 private:
  int16_t id_{-1};
  const char* label_{};
};

std::ostream& operator<<(std::ostream& os, const Node& node);

class ProgramVar final : public Node {
 public:
  ProgramVar(const Variable& var) : var_{var} {}
  const Variable* raw() const { return &var_; }

 private:
  Variable var_;
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
  bool Tell(const Node* var) const override;
  PatternVar* set_label(const char* label) {
    Node::set_label(label);
    return this;
  }

 private:
  bool external_{false};
  std::vector<std::function<bool(const Variable*)>> tellers_;
};

class PatternInstr final : public Node {
 public:
  PatternInstr(const char* type) : type_{type} {
    tellers_.emplace_back([=](const Instruction* instr) -> bool { return instr->get()->op_type == type_; });
  }
  const char* type() const { return type_; }
  PatternInstr* set_label(const char* label) {
    Node::set_label(label);
    return this;
  }

  bool Tell(const Node* instr) const override;

 private:
  const char* type_{};
  std::vector<std::function<bool(const Instruction*)>> tellers_;
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
  void Add(Node* start, Node* end, int16_t idx) { adj_[start].emplace(Target(end, idx)); }
  bool HasEdge(Node* start, Node* end) const;
  EdgeIterable edges() const { return EdgeIterable(*this); }

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
    auto* ret = nodes_.emplace(std::move(node)).first->get();
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

std::ostream& operator<<(std::ostream& os, const Digraph& graph);

class GraphBuilder {
 public:
  GraphBuilder()                    = default;
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder(GraphBuilder&&)      = default;
  virtual ~GraphBuilder()           = default;
  virtual Digraph release()         = 0;

 protected:
  Digraph graph_;
};

class PatternBuilder final : public GraphBuilder {
 public:
  PatternVar* AddVar();

  PatternInstr* AddInstr(const char* type,
                         const std::vector<PatternVar*>& inputs,
                         const std::vector<PatternVar*>& outputs);

  int16_t cur_id() const { return cur_id_; }
  Digraph release() override { return std::move(graph_); }

 private:
  int16_t cur_id_{-1};
};

class ProgramGraphBuilder final : public GraphBuilder {
 public:
  ProgramGraphBuilder(const Program& program);
  Digraph release() override { return std::move(graph_); }

 private:
  void AddInstr(const Instruction& instr);
  bool VarExists(const Variable& var) const { return var_map_.count(var.get()); }
  void AddVar(const Variable& var);

  int16_t cur_id_{-1};
  std::map<const _Variable_*, ProgramVar*> var_map_;
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

 private:
  class HitGroup {
   public:
    const std::map<pattern_node_t const*, program_node_t const*, NodeLessThan>& roles() const { return roles_; }
    void Register(program_node_t const* node, pattern_node_t const* pat) {
      roles_[pat] = node;
      nodes_.insert(node);
    }
    bool Match(program_node_t const* node, pattern_node_t const* pat) const;

   private:
    std::map<pattern_node_t const*, program_node_t const*, NodeLessThan> roles_;
    std::set<program_node_t const*> nodes_;
  };

  void NodeMatch();
  Digraph const* program_{};
  Digraph const* pattern_{};
  std::map<pattern_node_t*, std::vector<program_node_t*>, NodeLessThan> pdnodes2nodes_;
  EdgeIterable pattern_edges_;
  std::vector<HitGroup> groups_;
};

// TODO: rewrite the program efficiently
const cinn::frontend::Instruction* GetMatchedInstr(const PatternMatcher::pattern_map_t& matches, const char* label);
const cinn::frontend::Variable* GetMatchedVar(const PatternMatcher::pattern_map_t& matches, const char* label);

}  // namespace cinn::frontend::pass

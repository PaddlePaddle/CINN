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

#include "cinn/frontend/pass/pattern.h"

#include <iostream>

namespace cinn::frontend::pass {

std::ostream& operator<<(std::ostream& os, const Node& node) {
  std::string type;
  auto* instr = dynamic_cast<ProgramInstr const*>(&node);
  auto* var   = dynamic_cast<ProgramVar const*>(&node);
  if (instr) {
    type = instr->raw()->get()->op_type;
  } else if (var) {
    type = var->raw()->get()->id;
  }
  os << "Node (" << node.id() << ") " << type << " [" << &node << "]";
  return os;
}

bool PatternVar::Tell(const Node* var) const {
  bool ret          = true;
  const auto* p_var = dynamic_cast<ProgramVar const*>(var);
  if (p_var) {
    for (const auto& teller : tellers_) {
      ret = ret && teller(p_var->raw());
    }
  } else {
    ret = false;
  }
  return ret;
}

bool PatternInstr::Tell(const Node* instr) const {
  bool ret            = true;
  const auto* p_instr = dynamic_cast<ProgramInstr const*>(instr);
  if (p_instr) {
    for (const auto& teller : tellers_) {
      ret = ret && teller(p_instr->raw());
    }
  } else {
    ret = false;
  }
  return ret;
}

bool NodeLessThan::operator()(const Node* lhs, const Node* rhs) const {
  CHECK(lhs && rhs);
  return lhs->id() < rhs->id();
}

bool NodeLessThan::operator()(const std::pair<Node const*, Node const*>& lhs,
                              const std::pair<Node const*, Node const*>& rhs) const {
  bool res = false;
  if (lhs.first->id() < rhs.first->id()) {
    res = true;
  } else if (lhs.first->id() == rhs.first->id()) {
    res = lhs.second->id() < rhs.second->id();
  }
  return res;
}

bool Adjacent::HasEdge(Node* start, Node* end) const {
  if (!adj_.count(start)) {
    return false;
  }
  return adj_.at(start).count(Target(end, 0));
}

EdgeIterable::EdgeIterable(const Adjacent& adj) {
  for (auto& pair : adj.adj_) {
    for (auto& target : pair.second) {
      edges_.emplace(std::pair<Node*, Node*>(pair.first, target.end()));
    }
  }
}

bool operator<(const Target& lhs, const Target& rhs) { return lhs.end() < rhs.end(); }

PatternVar* PatternBuilder::AddVar() {
  auto var = std::make_unique<PatternVar>();
  var->set_id(++cur_id_);
  PatternVar* ret = var.get();
  graph_.AddNode(std::move(var));
  return ret;
}

PatternInstr* PatternBuilder::AddInstr(const char* type,
                                       const std::vector<PatternVar*>& inputs,
                                       const std::vector<PatternVar*>& outputs) {
  auto instr = std::make_unique<PatternInstr>(type);
  instr->set_id(++cur_id_);
  PatternInstr* ret = instr.get();
  graph_.AddNode(std::move(instr));

  for (size_t i = 0; i < inputs.size(); ++i) {
    graph_.AddEdge(inputs[i], ret, i);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    graph_.AddEdge(ret, outputs[i], i);
  }
  return ret;
}

std::ostream& operator<<(std::ostream& os, const Digraph& graph) {
  for (auto& a : graph.adj().edges()) {
    std::cout << *(a.first) << " -> " << *(a.second) << '\n';
  }
  return os;
}

ProgramGraphBuilder::ProgramGraphBuilder(const Program& program) {
  for (size_t i = 0; i < program.size(); ++i) {
    AddInstr(program[i]);
  }
}

void ProgramGraphBuilder::AddInstr(const Instruction& instr) {
  auto p_instr    = std::make_unique<ProgramInstr>(instr);
  auto* raw_instr = p_instr.get();
  p_instr->set_id(++cur_id_);
  graph_.AddNode(std::move(p_instr));

  for (size_t i = 0; i < instr->inputs.size(); ++i) {
    const auto& var = instr->inputs[i];
    if (!VarExists(var)) {
      AddVar(var);
    }
    graph_.AddEdge(var_map_.at(var.get()), raw_instr, i);
  }
  for (size_t i = 0; i < instr->outputs.size(); ++i) {
    const auto& var = instr->outputs[i];
    if (!VarExists(var)) {
      AddVar(var);
    }
    graph_.AddEdge(raw_instr, var_map_.at(var.get()), i);
  }
}

void ProgramGraphBuilder::AddVar(const Variable& var) {
  CHECK(!VarExists(var)) << "Repeated addition of variables is not allowed.";
  auto p_var = std::make_unique<ProgramVar>(var);
  auto* raw  = p_var.get();
  p_var->set_id(++cur_id_);
  graph_.AddNode(std::move(p_var));
  var_map_[var.get()] = raw;
}

void PatternMatcher::Init(const Digraph& pattern, const Digraph& program) {
  program_       = &program;
  pattern_       = &pattern;
  pattern_edges_ = pattern_->adj().edges();
  NodeMatch();
  VLOG(5) << "[Program Graph] " << program;
  VLOG(5) << "[Pattern Graph] " << pattern;
}

std::vector<PatternMatcher::pattern_map_t> PatternMatcher::DetectPatterns() {
  std::vector<PatternMatcher::pattern_map_t> res;
  std::array<std::vector<HitGroup>, 2> bi_records;
  auto& init_groups = bi_records[0];

  auto* first_pnode = pdnodes2nodes_.begin()->first;
  if (!pdnodes2nodes_.count(first_pnode)) {
    return res;
  }
  for (auto* node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.Register(node, first_pnode);
    init_groups.emplace_back(std::move(group));
  }
  int step{0};
  for (const auto& edge : pattern_edges_) {
    auto& pre_groups = bi_records[step % 2];
    auto& cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    for (auto* source : pdnodes2nodes_[edge.first]) {
      for (auto* target : pdnodes2nodes_[edge.second]) {
        for (const auto& group : pre_groups) {
          if (program_->adj().HasEdge(source, target)) {
            HitGroup new_group = group;
            bool flag          = new_group.Match(source, edge.first) && new_group.Match(target, edge.second);
            if (flag) {
              new_group.Register(source, edge.first);
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
            }
          }
        }
      }
    }
  }

  // TODO: Distinguishing and processing of external nodes.
  std::set<program_node_t const*> visited;
  for (auto& group : bi_records[step % 2]) {
    std::map<pattern_node_t const*, program_node_t const*> subgraph;
    bool overlapped{false};
    for (auto& role : group.roles()) {
      if (visited.find(role.second) == visited.end()) {
        subgraph.emplace(role.first, role.second);
      } else {
        overlapped = true;
      }
    }
    if (!overlapped) {
      for (auto& role : group.roles()) {
        visited.emplace(role.second);
      }
      VLOG(5) << "[Matched] : pattern -> program";
      for (auto& pair : subgraph) {
        VLOG(5) << "   -- " << *(pair.first) << " -> " << *(pair.second);
      }
      res.emplace_back(std::move(subgraph));
    }
  }
  return res;
}

bool PatternMatcher::HitGroup::Match(program_node_t const* node, pattern_node_t const* pat) const {
  if (nodes_.count(node)) {
    if (roles_.count(pat) && roles_.at(pat) == node) return true;
    return false;
  } else {
    if (roles_.count(pat) && roles_.at(pat) != node) return false;
    return true;
  }
}

void PatternMatcher::NodeMatch() {
  for (auto& pt_node : pattern_->nodes()) {
    for (auto& pr_node : program_->nodes()) {
      if (pt_node->Tell(pr_node.get())) {
        pdnodes2nodes_[pt_node.get()].emplace_back(pr_node.get());
      }
    }
  }
}

const cinn::frontend::Instruction* GetMatchedInstr(const PatternMatcher::pattern_map_t& matches, const char* label) {
  for (auto& match : matches) {
    if (!std::strcmp(label, match.first->label())) {
      const auto* program_node = dynamic_cast<ProgramInstr const*>(match.second);
      if (program_node) {
        return program_node->raw();
      }
    }
  }
  return {};
}

const cinn::frontend::Variable* GetMatchedVar(const PatternMatcher::pattern_map_t& matches, const char* label) {
  for (auto& match : matches) {
    if (!std::strcmp(label, match.first->label())) {
      const auto* var_node = dynamic_cast<ProgramVar const*>(match.second);
      if (var_node) {
        return var_node->raw();
      }
    }
  }
  return {};
}
}  // namespace cinn::frontend::pass

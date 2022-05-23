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

#include <algorithm>
#include <iterator>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/utils/type_defs.h"

namespace cinn::frontend::pass {

using cinn::utils::DimType;
using cinn::utils::ShapeType;

class MatmulKey {
 public:
  MatmulKey(Instruction* matmul) { SetKey(matmul); }

  void SetKey(Instruction* matmul) {
    shape_A = (*matmul)->inputs[0]->shape;
    shape_B = (*matmul)->inputs[1]->shape;

    const auto& attrs = (*matmul)->attrs;
    if (attrs.count("trans_a")) {
      trans_a = absl::get<bool>(attrs.at("trans_a"));
    } else {
      trans_a = false;
    }
    if (attrs.count("trans_b")) {
      trans_b = absl::get<bool>(attrs.at("trans_b"));
    } else {
      trans_b = false;
    }
    if (attrs.count("alpha")) {
      alpha = absl::get<float>(attrs.at("alpha"));
    } else {
      alpha = 1.0f;
    }
  }

  bool operator==(const MatmulKey& other) const {
    return shape_A == other.shape_A && shape_B == other.shape_B && trans_a == other.trans_a &&
           trans_b == other.trans_b && alpha == other.alpha;
  }
  bool operator!=(const MatmulKey& other) const { return !this->operator==(other); }

  std::string to_string() const {
    std::ostringstream hash_str;

    hash_str << "{A[" << cinn::utils::Join(shape_A, ",") << "], B[" << cinn::utils::Join(shape_B, ",")
             << "], trans_a:" << trans_a << ", trans_b:" << trans_b << ", alpha:" << alpha << "}";
    return hash_str.str();
  }

  struct Hash {
    size_t operator()(const MatmulKey& key) const {
      std::ostringstream hash_str;

      std::for_each(key.shape_A.begin(), key.shape_A.end(), [&](const DimType& dim) { hash_str << dim; });
      std::for_each(key.shape_B.begin(), key.shape_B.end(), [&](const DimType& dim) { hash_str << dim; });

      hash_str << key.trans_a;
      hash_str << key.trans_b;
      hash_str << key.alpha;

      return std::hash<std::string>()(hash_str.str());
    }
  };

  ShapeType shape_A, shape_B;
  bool trans_a, trans_b;
  float alpha;
};

class MatmulMergePass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  using OutputToOpMap    = std::unordered_map<std::string, Instruction*>;
  using InputToOpMap     = std::unordered_map<std::string, std::unordered_set<Instruction*>>;
  using OpToAncestorsMap = std::unordered_map<Instruction*, std::unordered_set<Instruction*>>;
  using MatmulKeyMap     = std::unordered_map<MatmulKey, std::vector<Instruction*>, MatmulKey::Hash>;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    VLOG(3) << "-- Before MatmulMergePass:\n" << *program;
    std::vector<Instruction*> all_instrs;
    // `out2instr` is used to represent the mapping of Output to Instruction.
    OutputToOpMap out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    InputToOpMap in2instr;
    // the same matmul op will gather together
    // in detail, the matmul same means the value of A.shape,B.shape,trans_a,trans_b,alpha are all same
    MatmulKeyMap same_matmul_map;

    for (size_t i = 0; i < program->size(); ++i) {
      auto instr = &((*program)[i]);

      all_instrs.emplace_back(instr);

      for (const auto& out : (*instr)->outputs) {
        out2instr[out->id] = instr;
      }
      for (const auto& in : (*instr)->inputs) {
        in2instr[in->id].insert(instr);
      }

      if ("matmul" == (*instr)->op_type) {
        MatmulKey key(instr);
        same_matmul_map[key].emplace_back(instr);
      }
    }

    VLOG(4) << "Step1: Get entrance instructions";
    auto out_instrs = GetGraphOutInstrs(all_instrs, in2instr);

    VLOG(4) << "Step2: Get each instruction's ancestor nodes list";
    auto no_link_matmul_map = GetNoLinkMatmulNodeMap(all_instrs, out2instr, out_instrs);

    VLOG(4) << "Step3: Get the matmul group of whose inputs shape are all same and no ancestor relationship";
    auto groups = GetFusionGroups(same_matmul_map, no_link_matmul_map);

    // CinnBuilder builder("matmul_merge_builder");
    // for (auto& var : program->GetInputs()) {
    //   builder.CreateInput(var);
    // }

    // std::unordered_set<Instruction*> remove_instrs;

    // for (int i = 0; i < program->size(); i++) {
    //   if (remove_instrs.end() != remove_instrs.find(&(*program)[i])) continue;
    //   builder.AppendInstruction((*program)[i]);
    // }
    // *program = builder.Build();

    VLOG(3) << "-- After MatmulMergePass:\n" << *program;
  }

  // get the entrance instruction, whose inputs are all from out-graph
  std::unordered_set<Instruction*> GetGraphOutInstrs(const std::vector<Instruction*>& all_instrs,
                                                     const InputToOpMap& in2instr) const {
    std::unordered_set<Instruction*> out_instrs;
    for (size_t i = 0; i < all_instrs.size(); ++i) {
      auto instr   = all_instrs[i];
      bool is_exit = true;
      for (const auto& out : (*instr)->outputs) {
        if (in2instr.count(out->id)) {
          // if the instruction's input are some instruction's output, it's not entrance.
          is_exit = false;
          break;
        }
      }
      if (is_exit) {
        out_instrs.insert(instr);
      }
    }

    std::string res;
    for (auto instr : out_instrs) {
      res += (*instr)->op_type + ", ";
    }
    VLOG(3) << "Graph's output instruction: " << res;
    return out_instrs;
  }

  // get each instruction's ancestor list
  OpToAncestorsMap GetAllAncestorNodeMap(const OutputToOpMap& out2instr,
                                         const std::unordered_set<Instruction*>& out_instrs) const {
    OpToAncestorsMap ancestor_map;
    std::unordered_set<Instruction*> visited_instr;

    std::queue<Instruction*> next_instr;
    for (auto instr : out_instrs) {
      next_instr.push(instr);
      // create a empty ancestor list
      ancestor_map[instr].clear();
    }

    // level traversal
    while (!next_instr.empty()) {
      auto cur_instr = next_instr.front();
      next_instr.pop();

      if (visited_instr.count(cur_instr)) continue;
      visited_instr.insert(cur_instr);

      for (const auto& in : (*cur_instr)->inputs) {
        if (!out2instr.count(in->id)) continue;

        auto in_instr = out2instr.at(in->id);
        // the input instr's son nodes contain the current node and its son nodes
        ancestor_map[in_instr].insert(cur_instr);
        ancestor_map[in_instr].insert(ancestor_map.at(cur_instr).begin(), ancestor_map.at(cur_instr).end());

        if (!visited_instr.count(in_instr)) {
          // if not visited, push into queue for traversal
          next_instr.push(in_instr);
        }
      }
    }
    return ancestor_map;
  }

  OpToAncestorsMap GetMatmulAncestorNodeMap(const std::unordered_set<Instruction*>& not_matmul_set,
                                            const OutputToOpMap& out2instr,
                                            const std::unordered_set<Instruction*>& out_instrs) const {
    auto ancestor_map = GetAllAncestorNodeMap(out2instr, out_instrs);

    for (auto instr : not_matmul_set) {
      ancestor_map.erase(instr);
    }

    for (const auto& instr_pair : ancestor_map) {
      for (auto instr : not_matmul_set) {
        instr_pair.second.erase(instr);
      }
    }

    auto debug_info = [](const OpToAncestorsMap& node_map) {
      std::string res;
      for (const auto& pair : node_map) {
        res += (*pair.first)->op_type + "/" + pair.first->GetOutput(0)->id + ": ";
        for (auto instr : pair.second) {
          res += (*instr)->op_type + "/" + instr->GetOutput(0)->id + ", ";
        }
        res += "\n";
      }
      return res;
    };
    VLOG(5) << "Matmul and its ancestor list:\n" << debug_info(ancestor_map);

    return ancestor_map;
  }

  OpToAncestorsMap GetNoLinkMatmulNodeMap(const std::vector<Instruction*>& all_instrs,
                                          const OutputToOpMap& out2instr,
                                          const std::unordered_set<Instruction*>& out_instrs) const {
    std::unordered_set<Instruction*> matmul_set, not_matmul_set;
    for (auto instr : all_instrs) {
      if ("matmul" != (*instr)->op_type) {
        not_matmul_set.insert(instr);
        continue;
      } else {
        matmul_set.insert(instr);
      }
    }

    auto ancestor_map = GetMatmulAncestorNodeMap(not_matmul_set, out2instr, out_instrs);

    OpToAncestorsMap no_link_matmul_map;
    for (const auto& instr_pair : ancestor_map) {
      std::set_difference(matmul_set.begin(),
                          matmul_set.end(),
                          instr_pair.second.begin(),
                          instr_pair.second.end(),
                          std::inserter(no_link_matmul_map[instr_pair.first].begin()));
    }

    auto debug_info = [](const OpToAncestorsMap& node_map) {
      std::string res;
      for (const auto& pair : node_map) {
        res += (*pair.first)->op_type + "/" + pair.first->GetOutput(0)->id + ": ";
        for (auto instr : pair.second) {
          res += (*instr)->op_type + "/" + instr->GetOutput(0)->id + ", ";
        }
        res += "\n";
      }
      return res;
    };
    VLOG(5) << "Matmul and its not-link brother matmul list:\n" << debug_info(no_link_matmul_map);

    return no_link_matmul_map;
  }

  std::vector<std::vector<Instruction*>> GetFusionGroups(const MatmulKeyMap& same_matmul_map,
                                                         const OpToAncestorsMap& no_link_matmul_map) const {
    auto debug_output_names = [](const std::vector<Instruction*>& group) -> std::string {
      std::string output_names;
      for (auto instr : group) {
        output_names += instr->GetOutput(0)->id + ", ";
      }
      return output_names;
    };

    std::vector<std::vector<Instruction*>> fusion_groups;
    std::unordered_set<Instruction*> instrs_has_group;
    for (const auto& instr_pair : same_matmul_map) {
      // find group in the list of same matmul
      instrs_has_group.clear();

      const auto& candidate = instr_pair.second;
      for (int i = 0; i < candidate.size() - 1; ++i) {
        if (instrs_has_group.count(candidate[i])) continue;

        // try find the group of candidate[i], of course, candidate[i] in its group
        fusion_groups.emplace_back(std::vector<Instruction*>{candidate[i]});
        instrs_has_group.insert(candidate[i]);

        for (int j = i + 1; j < candidate.size(); ++j) {
          if (instrs_has_group.count(candidate[j]) || is_ancestor(candidate[i], candidate[j])) continue;

          // they are not link, can fuse
          fusion_groups.back().emplace_back(candidate[j]);
          // the instruction is in group now
          instrs_has_group.insert(candidate[j]);
        }

        VLOG(3) << "Group of " << instr_pair.first.to_string() << " size is " << fusion_groups.back().size();
        VLOG(4) << "The output of each matmul in group are: " << debug_output_names(fusion_groups.back());
      }
    }
    return fusion_groups;
  }

  void FuseGroup(CinnBuilder* builder, const std::vector<Instruction*>& group) const {
    bool is_reshaped = false;

    std::vector<Variable> matrix_A_list, matrix_B_list, matrix_C_list;
    for (auto matmul : group) {
      const auto& matrix_A = (*matmul)->inputs[0];
      const auto& matrix_B = (*matmul)->inputs[1];
      const auto& matrix_C = (*matmul)->outputs[0];

      auto a_shape_dim = matrix_A->shape.size();
      auto b_shape_dim = matrix_B->shape.size();
      auto c_shape_dim = matrix_C->shape.size();

      CHECK_EQ(a_shape_dim, b_shape_dim) << "The dimension of matmul matrix A and B should be the same, but "
                                         << matrix_A->id << " and " << matrix_B->id << " not ! Please check.";
      CHECK_EQ(a_shape_dim, c_shape_dim) << "The dimension of matmul matrix A and C should be the same, but "
                                         << matrix_A->id << " and " << matrix_C->id << " not ! Please check.";
      CHECK(a_shape_dim >= 2 && a_shape_dim <= 3) << "The dimension of matmul matrix should be 2 or 3, but matrix "
                                                  << matrix_A->id << "'s dimension is " << a_shape_dim;

      if (matrix_A->shape.size() == 2UL) {
        is_reshaped = true;

        auto new_shape_A = matrix_A->shape;
        new_shape_A.insert(new_shape_A.begin(), 1);
        auto reshaped_matrix_A = builder->Reshape(matrix_A, new_shape_A);
        matrix_A_list.emplace_back(reshaped_matrix_A);

        auto new_shape_B = matrix_B->shape;
        new_shape_B.insert(new_shape_B.begin(), 1);
        auto reshaped_matrix_B = builder->Reshape(matrix_B, new_shape_B);
        matrix_B_list.emplace_back(reshaped_matrix_B);
      } else {
        matrix_A_list.emplace_back(matrix_A);
        matrix_B_list.emplace_back(matrix_B);
      }

      matrix_C_list.emplace_back(matrix_C);
    }

    auto concated_A = builder->Concat(matrix_A_list, 0);
    auto concated_B = builder->Concat(matrix_B_list, 0);

    // auto C = builder->CustomInstr("matmul", std::vector<Variable>{concated_A, concated_B});
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(MatmulMerge) {
  CINN_REGISTER_PROGRAM_PASS(MatmulMerge, ::cinn::frontend::pass::MatmulMergePass);

  return true;
}

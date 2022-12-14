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

#include <glog/logging.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cinn {
namespace auto_schedule {

std::vector<std::string> SplitString(const std::string& str, const char split) {
  std::stringstream ss(str);
  std::string token;
  std::vector<std::string> res;
  while (getline(ss, token, split)) {
    res.push_back(token);
  }
  return res;
}

std::pair<std::string, std::vector<int32_t>> ParseVarNameAndShape(const std::string& line) {
  // parse var name
  std::regex var_name_pattern("var (.+) :");
  std::smatch var_name_match;
  std::regex_search(line, var_name_match, var_name_pattern);
  VLOG(6) << "parse variable name: " << var_name_match[1];

  // parse var shape
  std::regex shape_pattern("LOD_TENSOR.shape\\((.+)\\).dtype");
  std::smatch shape_match;
  std::regex_search(line, shape_match, shape_pattern);
  VLOG(6) << "parse shape: " << shape_match[1];
  std::vector<std::string> shape_strs = SplitString(shape_match[1], ',');
  std::vector<int32_t> shape;
  std::transform(shape_strs.begin(), shape_strs.end(), std::back_inserter(shape), [](const std::string& str) {
    return stoi(str);
  });

  return std::make_pair(var_name_match[1], shape);
}

std::pair<std::string, std::string> ParseOpNameAndInfo(const std::string& line) {
  std::regex pattern("\\{.+\\} = (.+)\\((.+)\\)");
  std::smatch match;
  std::regex_search(line, match, pattern);
  VLOG(6) << "parse op name: " << match[1];
  VLOG(6) << "parse op info: " << match[2];

  return std::make_pair(match[1], match[2]);
}

std::vector<std::string> ParseInputVarNames(const std::string& str) {
  std::regex pattern("inputs=\\{(.+)\\}");
  std::smatch match;
  std::regex_search(str, match, pattern);
  std::vector<std::string> input_strs = SplitString(match[1], ',');
  std::vector<std::string> input_var_names;
  std::transform(input_strs.begin(), input_strs.end(), std::back_inserter(input_var_names), [](const std::string& s) {
    std::regex pattern(".+=\\['(.+)'\\]");
    std::smatch match;
    std::regex_search(s, match, pattern);
    VLOG(7) << "input_var_name: " << match[1];
    return match[1];
  });

  return input_var_names;
}

std::tuple<std::string, std::string> ParseMatmulTransposeInfo(const std::string& str) {
  std::regex pattern("trans_x = (.+), trans_y = (.+)");
  std::smatch match;
  std::regex_search(str, match, pattern);
  VLOG(7) << "trans_x: " << match[1] << ", trans_y: " << match[2];
  return std::make_tuple(match[1], match[2]);
}

struct MatmulInfo {
  std::string name;
  std::vector<int32_t> shape_x;
  std::vector<int32_t> shape_y;
  bool trans_x;
  bool trans_y;

  MatmulInfo(const std::string& p_name,
             const std::vector<int32_t>& p_shape_x,
             const std::vector<int32_t>& p_shape_y,
             bool p_trans_x,
             bool p_trans_y)
      : name(p_name), shape_x(p_shape_x), shape_y(p_shape_y), trans_x(p_trans_x), trans_y(p_trans_y) {
    bool is_same_position = trans_x ^ trans_y;

    if (shape_x.size() - 1 >= 0 && shape_x.at(shape_x.size() - 1) == -1) {
      shape_x[shape_x.size() - 1] = is_same_position ? shape_y[shape_y.size() - 1] : shape_y[shape_y.size() - 2];
    }
    if (shape_x.size() - 2 >= 0 && shape_x.at(shape_x.size() - 2) == -1) {
      shape_x[shape_x.size() - 2] = is_same_position ? shape_y[shape_y.size() - 2] : shape_y[shape_y.size() - 1];
    }
    if (shape_y.size() - 1 >= 0 && shape_y.at(shape_y.size() - 1) == -1) {
      shape_y[shape_y.size() - 1] = is_same_position ? shape_x[shape_x.size() - 1] : shape_x[shape_x.size() - 2];
    }
    if (shape_y.size() - 2 >= 0 && shape_y.at(shape_y.size() - 2) == -1) {
      shape_y[shape_y.size() - 2] = is_same_position ? shape_x[shape_x.size() - 2] : shape_x[shape_x.size() - 1];
    }
  }

  std::string DebugString() const {
    std::stringstream ss;
    ss << name << ": shape_x=[";
    for (int32_t shape : shape_x) {
      ss << shape << ",";
    }
    ss << "], shape_y=[";
    for (int32_t shape : shape_y) {
      ss << shape << ",";
    }
    ss << "], trans_x=" << trans_x << ", trans_y=" << trans_y;
    return ss.str();
  }
};

std::unordered_map<std::string, MatmulInfo> LoadMatmulInfo(const std::string& flie_path) {
  std::ifstream in_file(flie_path);
  if (!in_file.good()) {
    LOG(ERROR) << "open flie failed, flie_path: " << flie_path;
  }
  std::unordered_map<std::string, MatmulInfo> matmul_infos;
  std::unordered_map<std::string, std::vector<int32_t>> var_shape_map;
  std::string line;
  while (getline(in_file, line)) {
    // var line
    if (line.size() > 4 && (line.substr(4, 3) == "var" || line.substr(4, 3) == "per")) {
      var_shape_map.insert(ParseVarNameAndShape(line));
    }
    // op line
    if (line.size() > 4 && line.substr(4, 4) == "{Out") {
      auto op_name_and_info = ParseOpNameAndInfo(line);
      std::string op_name   = op_name_and_info.first;
      if (op_name != "matmul_v2") continue;
      std::vector<std::string> input_var_names = ParseInputVarNames(op_name_and_info.second);
      CHECK_EQ(input_var_names.size(), 2);
      std::vector<std::vector<int32_t>> shapes;
      for (const auto& var_name : input_var_names) {
        if (var_shape_map.count(var_name) == 0) {
          LOG(FATAL) << "Invalid var name: " << var_name << ",  op: " << op_name;
        }
        shapes.push_back(var_shape_map.at(var_name));
      }
      std::tuple<std::string, std::string> trans_info = ParseMatmulTransposeInfo(op_name_and_info.second);
      bool trans_x                                    = std::get<0>(trans_info) == "False" ? false : true;
      bool trans_y                                    = std::get<1>(trans_info) == "False" ? false : true;

      MatmulInfo info(op_name, shapes.at(0), shapes.at(1), trans_x, trans_y);
      matmul_infos.insert(std::make_pair(info.DebugString(), info));
    }
  }
  return matmul_infos;
}

}  // namespace auto_schedule
}  // namespace cinn

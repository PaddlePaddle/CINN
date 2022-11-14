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

#include "cinn/backends/nvrtc/header_generator.h"

#include "glog/logging.h"
#include "jitify.hpp"

namespace cinn {
namespace backends {
namespace nvrtc {
namespace {
#ifdef NVRTC_STL_PATH
static constexpr char* nvrtc_stl_path = NVRTC_STL_PATH;
#else
static constexpr char* nvrtc_stl_path = nullptr;
#endif
}  // namespace

JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  for (const auto& pair : headers_map_) {
    header_names_.emplace_back(pair.first);
  }
}

JitSafeHeaderGenerator::JitSafeHeaderGenerator(std::vector<std::string> header_names)
    : header_names_{std::move(header_names)} {}

void JitSafeHeaderGenerator::GenerateFiles(const std::string& dir) const {
  std::ofstream os;
  std::ifstream is;
  // TODO(Shixiaowei02): parallel execution
  for (const auto& name : header_names_) {
    std::string full_path = dir + "/" + name;
    is.open(full_path);
    if (is.good()) {
      LOG(INFO) << "file exists: " << full_path;
      is.close();
      continue;
    }
    is.close();
    os.open(full_path, std::ios_base::out);
    os << headers_map_.at(name);
    LOG(INFO) << "write file: " << full_path;
    os.close();
  }
}

void JitSafeHeaderGenerator::GenerateFiles() const {
  CHECK(nvrtc_stl_path) << "The default directory path does not exist.";
  GenerateFiles(nvrtc_stl_path);
}

const std::map<std::string, std::string> JitSafeHeaderGenerator::headers_map_ =
    jitify::detail::get_jitsafe_headers_map();

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn

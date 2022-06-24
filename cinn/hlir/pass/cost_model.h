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

#include <memory>
#include <string>
#include <vector>

#include "cinn/auto_schedule/cost_model/cost_model.h"
#include "cinn/hlir/framework/graph.h"

namespace cinn {
namespace hlir {
namespace pass {

using namespace framework;
using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

class FusionGroupComparator {
 public:
  FusionGroupComparator(const std::string& model_type, const std::string& model_path = "");
  float Predict(const GroupPtr& producer, const GroupPtr& consumer, const GroupPtr& fusion);

  void Train(const std::vector<GroupList>& src, const std::vector<float>& labels);
  void SaveModel(const std::string& model_path);

 private:
  struct Device {
    // device memory band width.
    int Band_Width;
    // device max parallel threads.
    int Max_Parallel;
    // device max clock.
    int Max_Clock;
  };
  /*
  1. element-wise fuse reduce.
  2. element-wise fuse broadcast.
  3. producer recompute.
  */
  // kernel feature
  struct KernelFeature {
    // the number of ops.
    int num_ops;
    // the size of tensor
    int num_elements;
    // the times of reader io
    int num_reader_ios;
    // the times of writer io
    int num_writer_ios;
    // the size of parallel.
    int parallel_size;

    KernelFeature operator+(const GroupPtr& others) {
      KernelFeature n_kf;
      return n_kf;
    }

    void operator+=(const GroupPtr& others) {}

    std::vector<float> GetModelFeature(const Device& device) {
      std::vector<float> feature;
      return feature;
    }
  };

  // extract kernel feature.
  KernelFeature ExtractKernelFeature(const GroupPtr& group) {
    KernelFeature kf;
    return kf;
  }

  std::unique_ptr<auto_schedule::CostModel> cost_model_;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

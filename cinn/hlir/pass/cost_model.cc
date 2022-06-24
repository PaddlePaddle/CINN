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

#include "cinn/hlir/pass/cost_model.h"

namespace cinn {
namespace hlir {
namespace pass {

FusionGroupComparator::FusionGroupComparator(const std::string& model_type, const std::string& model_path) {
  cost_model_.reset(new auto_schedule::CostModel());
  if (model_path.size()) {
    cost_model_->Load(model_path);
  } else {
    cost_model_->Load("");
  }
}

std::vector<float> operator+(const std::vector<float>& first, const std::vector<float>& second) {
  std::vector<float> res(first);
  res.insert(res.end(), second.begin(), second.end());
  return res;
}

float FusionGroupComparator::Predict(const GroupPtr& producer, const GroupPtr& consumer, const GroupPtr& fusion) {
  auto producer_kf = ExtractKernelFeature(producer);
  auto consumer_kf = ExtractKernelFeature(consumer);
  auto fusion_kf   = ExtractKernelFeature(fusion);

  Device device;
  auto produer_feature  = producer_kf.GetModelFeature(device);
  auto consumer_feature = consumer_kf.GetModelFeature(device);
  auto fusion_feature   = fusion_kf.GetModelFeature(device);

  auto pred = cost_model_->Predict({produer_feature + consumer_feature + fusion_feature});
  return pred[0];
}

void FusionGroupComparator::Train(const std::vector<GroupList>& groups_list, const std::vector<float>& labels) {
  Device device;
  std::vector<std::vector<float>> n_features;
  for (auto& groups : groups_list) {
    std::vector<float> feature;
    for (auto& group : groups) {
      auto kf = ExtractKernelFeature(group);
      feature = std::move(feature + kf.GetModelFeature(device));
    }
    n_features.push_back(std::move(feature));
  }
  cost_model_->Train(n_features, labels);
}

void FusionGroupComparator::SaveModel(const std::string& model_path) { cost_model_->Save(model_path); }

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
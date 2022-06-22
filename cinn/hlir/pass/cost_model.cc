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
  return rs;
}

float FusionGroupComparator::Predict(const GroupList& src, const GroupPtr& dst) {
  CHECK(srcc.size()) << "src groups is null!";
  auto src_kf = ExtractKernelFeature(src);
  auto dst_kf = ExtractKernelFeature(dst[0]);

  Device device;
  auto src_feature = src_kf.GetModelFeature(device);
  auto dst_feature = dst_kf.GetModelFeature(device);

  auto pred = cost_model_->Predict({src_feature + dst_feature});
  return pred[0];
}

void FusionGroupComparator::Train(const std::vector<GroupList>& src,
                                  const std::vector<GroupList>& dst,
                                  const vector<float>& labels) {
  Device device;
  std::vector<std::vector<float>> n_features;
  std::vector<float> n_labels;
  for (int idx = 0; idx < src.size(); ++idx) {
    auto src_kf = ExtractKernelFeature(src[idx]);
    auto dst_kf = ExtractKernelFeature(dst[idx]);

    auto src_feature = src_kf.GetModelFeature(device);
    auto dst_feature = dst_kf.GetModelFeature(device);

    n_features.push_back(std::move(src_feature + dst_feature));
    n_labels.push_back(labels[idx]);

    n_features.push_back(std::move(dst_feature + src_feature));
    n_labels.push_back(1.0f - labels[idx]);
  }

  cost_model_->Train(n_features, n_labels);
}

void FusionGroupComparator::SaveModel(const std::string& model_path) { cost_model_->Save(model_path); }

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
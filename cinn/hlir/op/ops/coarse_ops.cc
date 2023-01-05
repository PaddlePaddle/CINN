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

namespace cinn {
namespace hlir {
namespace op {

std::vector<shape_t> InferShapeForElementwise(const std::vector<shape_t> &inputs_shape,
                                              const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwise(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<shape_t> InferShapeForBatchNorm(const std::vector<shape_t> &inputs_shape,
                                            const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForBatchNorm(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<framework::shape_t> InferShapeForBatchNormTrain(const std::vector<framework::shape_t> &inputs_shape,
                                                            const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 5U) << "The input's layout size is not 5! Please check again.";
  std::string data_layout = "";
  if (attrs.find("data_layout") != attrs.end()) {
    data_layout = absl::get<std::string>(attrs.at("data_layout"));
  } else {
    LOG(FATAL) << "data_layout is not found, please check!";
  }

  CHECK_EQ(inputs_shape[0].size(), 4) << "x dimension size is not required!";
  CHECK_EQ(inputs_shape[1].size(), 1) << "scale dimension size is not required!";
  CHECK_EQ(inputs_shape[2].size(), 1) << "bias dimension size is not required!";
  CHECK_EQ(inputs_shape[3].size(), 1) << "moving_mean dimension size is not required!";
  CHECK_EQ(inputs_shape[4].size(), 1) << "moving_variance dimension size is not required!";

  if (data_layout == "NCHW") {
    CHECK_EQ(inputs_shape[0][1], inputs_shape[1][0]) << "x and scale dimension is not equal!";
    CHECK_EQ(inputs_shape[0][1], inputs_shape[2][0]) << "x and bias dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][1], inputs_shape[3][0]) << "x and moveing_mean dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][1], inputs_shape[4][0]) << "x and moveing_variance dimension size is not equal!";
  } else if (data_layout == "NHWC") {
    CHECK_EQ(inputs_shape[0][3], inputs_shape[1][0]) << "x and scale dimension is not equal!";
    CHECK_EQ(inputs_shape[0][3], inputs_shape[2][0]) << "x and bias dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][3], inputs_shape[3][0]) << "x and moveing_mean dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][3], inputs_shape[4][0]) << "x and moveing_variance dimension size is not equal!";
  } else {
    LOG(FATAL) << "data_layout " << data_layout << " is not support!";
  }

  return {inputs_shape[0], inputs_shape[1], inputs_shape[1], inputs_shape[1], inputs_shape[1]};
}

std::vector<Type> InferDtypeForBatchNormTrain(const std::vector<Type> &inputs_type,
                                              const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {inputs_type[0], inputs_type[0], inputs_type[0], inputs_type[0], inputs_type[0]};
}

std::vector<framework::shape_t> InferShapeForBatchNormGrad(const std::vector<framework::shape_t> &inputs_shape,
                                                           const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 5U) << "The input's layout size is not 5! Please check again.";
  std::string data_layout = "";
  if (attrs.find("data_layout") != attrs.end()) {
    data_layout = absl::get<std::string>(attrs.at("data_layout"));
  } else {
    LOG(FATAL) << "data_layout is not found, please check!";
  }

  CHECK_EQ(inputs_shape[0].size(), 4) << "dy dimension size is not required!";
  CHECK_EQ(inputs_shape[1].size(), 4) << "x dimension size is not required!";
  CHECK_EQ(inputs_shape[2].size(), 1) << "scale dimension size is not required!";
  CHECK_EQ(inputs_shape[3].size(), 1) << "save_mean dimension size is not required!";
  CHECK_EQ(inputs_shape[4].size(), 1) << "save_variance dimension size is not required!";

  CHECK(inputs_shape[0] == inputs_shape[1]) << "dy and x shape is not equal!";
  if (data_layout == "NCHW") {
    CHECK_EQ(inputs_shape[0][1], inputs_shape[2][0]) << "dy and bias dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][1], inputs_shape[3][0]) << "dy and moveing_mean dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][1], inputs_shape[4][0]) << "dy and moveing_variance dimension size is not equal!";
  } else if (data_layout == "NHWC") {
    CHECK_EQ(inputs_shape[0][3], inputs_shape[2][0]) << "dy and bias dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][3], inputs_shape[3][0]) << "dy and moveing_mean dimension size is not equal!";
    CHECK_EQ(inputs_shape[0][3], inputs_shape[4][0]) << "dy and moveing_variance dimension size is not equal!";
  } else {
    LOG(FATAL) << "data_layout " << data_layout << " is not support!";
  }

  return {inputs_shape[0], inputs_shape[2], inputs_shape[2]};
}

std::vector<Type> InferDtypeForBatchNormGrad(const std::vector<Type> &inputs_type,
                                             const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {inputs_type[0], inputs_type[0], inputs_type[0]};
}

std::vector<framework::shape_t> InferShapeForConv2dGrad(const std::vector<framework::shape_t> &inputs_shape,
                                                        const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 3U) << "The input's layout size is not 3! Please check again.";
  CHECK_EQ(inputs_shape[0].size(), 4U) << "Dy shape is not 4, Please check again.";
  CHECK_EQ(inputs_shape[1].size(), 4U) << "Dy shape is not 4, Please check again.";
  CHECK_EQ(inputs_shape[2].size(), 4U) << "Dy shape is not 4, Please check again.";
  return {inputs_shape[1], inputs_shape[2]};
}

std::vector<Type> InferDtypeForConv2dGrad(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {inputs_type[0], inputs_type[0]};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(coarse_ops) {
  CINN_REGISTER_OP(relu)
      .describe("The implements of relu.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForELementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(relu_grad)
      .describe("The gradient of relu.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForELementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise));

  CINN_REGISTER_OP(batch_norm)
      .describe("The implements of batch_norm.")
      .set_num_inputs(5)  // batchnorm(mean, variance, scale, bias)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBatchNorm))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBatchNorm));

  CINN_REGISTER_OP(batch_norm_train)
      .describe("This operator implements the batch normalization training forward.")
      .set_num_inputs(5)
      .set_num_outputs(5)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBatchNormTrain))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBatchNormTrain));

  CINN_REGISTER_OP(batch_norm_grad)
      .describe("This operator implements the batch normalization backward.")
      .set_num_inputs(5)
      .set_num_outputs(3)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBatchNormGrad))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBatchNormGrad));

  CINN_REGISTER_OP(conv2d_grad)
      .describe("This operator implements the convolution backward.")
      .set_num_inputs(3)
      .set_num_outputs(2)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForConv2dGrad))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForConv2dGrad));

  CINN_REGISTER_OP(softmax)
      .describe("This operator implements the softmax layer")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForELementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise));

  CINN_REGISTER_OP(gelu)
      .describe("The implement of gelu.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForELementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise));

  CINN_REGISTER_OP(clip)
      .describe("The implement of clip.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise));

  return true;
}

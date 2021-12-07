// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

//! @h1 Load and Execute Paddle Model in C++
//! In this tutorial, we will show you how to load and execute a paddle model in CINN using C++.
//! We will use model ResNet18 as an example.

#include <gtest/gtest.h>

#include "cinn/cinn.h"

using namespace cinn;  // NOLINT

//! @IGNORE-NEXT
TEST(LOAD_MODEL, basic) {
  //! @h2 Prepare to Load Model
  //! Declare the params and prepare to load and execute the paddle model.
  //! + `input_name` is the name of input tensor in the model.
  //! + `target_name` is the name of output tensor we want.
  //! + `x_shape` is the input tensor's shape of the model.

  std::string input_name   = "image";
  std::string target_name  = "save_infer_model/scale_0";
  std::vector<int> x_shape = {1, 3, 224, 224};

  //! @h2 Set the target backend
  //! Now CINN only supports two backends: X86 and CUDA.
  //! + To choose X86 backends, use :
  //! `auto target = common::DefaultHostTarget();`
  //! + To choose CUDA backends, use :
  //! `auto target = common::DefaultNVGPUTarget();`

  auto target = common::DefaultHostTarget();

  //! @h2 Set the input tensor and init interpreter
  frontend::Interpreter executor({input_name}, {x_shape});

  //! @h2 Load Model to CINN
  //! Load the paddle model and transform it into CINN IR.
  //! + `mnodel_dir` is the path where the paddle model is stored.
  //! + `target` is the backend to execute model on.
  //! + `params_combined` implies whether the params of paddle model is stored in one file.
  //! + `model_name` is the name of the model. Entering this will enable optimizations for each specific model.

  std::string model_dir  = "./ResNet18";
  std::string model_name = "resnet18";
  bool params_combined   = true;
  executor.LoadPaddleModel(model_dir, target, params_combined, model_name);

  //! @h2 Get input tensor and set input data
  //! Here we use all-zero data as input. In practical applications, please replace it with real data according to your
  //! needs.

  auto input_tensor = executor.GetTensor(input_name);

  std::vector<float> fake_input(input_tensor->shape().numel(), 0.f);

  auto *input_data = input_tensor->mutable_data<float>(target);
  if (target.arch == Target::Arch::X86) {
    std::copy(fake_input.begin(), fake_input.end(), input_data);
  } else if (target.arch == Target::Arch::NVGPU) {
    CUDA_CALL(cudaMemcpy(
        input_data, fake_input.data(), input_tensor->shape().numel() * sizeof(float), cudaMemcpyHostToDevice));
  }

  //! @h2 Execute Model
  //! Execute the model and get output tensor's data.

  executor.Run();

  auto target_tensor = executor.GetTensor(target_name);
  std::vector<float> output_data(target_tensor->shape().numel(), 0.f);
  if (target.arch == Target::Arch::X86) {
    std::copy(target_tensor->data<float>(),
              target_tensor->data<float>() + target_tensor->shape().numel(),
              output_data.data());
  } else if (target.arch == Target::Arch::NVGPU) {
    CUDA_CALL(cudaMemcpy(output_data.data(),
                         reinterpret_cast<void *>(target_tensor->mutable_data<float>(target)),
                         target_tensor->shape().numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }
  //! @IGNORE-NEXT
  LOG(INFO) << "Succeed!";
}

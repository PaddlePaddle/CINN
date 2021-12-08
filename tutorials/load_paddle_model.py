# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Load and Execute Paddle Model
=====================

In this tutorial, we will show you how to load and execute a paddle model in CINN.
We offer you four optional models: ResNet50, MobileNetV2, EfficientNet and FaceDet.
"""

import cinn
from cinn import *
from cinn.frontend import *
from cinn.framework import *
from cinn.common import *
import numpy as np
import os

##################################################################
# **Prepare to Load Model**
# --------------------------
# Declare the params and prepare to load and execute the paddle model.
#
# - :code:`model_dir` is the path where the paddle model is stored.
#
# - :code:`input_tensor` is the name of input tensor in the model.
#
# - :code:`target_tensor` is the name of output tensor we want.
#
# - :code:`x_shape` is the input tensor's shape of the model
#
# - When choosing model ResNet50, the params should be ::
#
#       model_dir = "./ResNet50"
#
#       input_tensor = 'inputs'
#
#       target_tensor = 'save_infer_model/scale_0.tmp_1'
#
#       x_shape = [1, 3, 224, 224]
#
# - When choosing model MobileNetV2, the params should be ::
#
#       model_dir = "./MobileNetV2"
#
#       input_tensor = 'image'
#
#       target_tensor = 'save_infer_model/scale_0'
#
#       x_shape = [1, 3, 224, 224]
#
# - When choosing model EfficientNet, the params should be ::
#
#       model_dir = "./EfficientNet"
#
#       input_tensor = 'image'
#
#       target_tensor = 'save_infer_model/scale_0'
#
#       x_shape = [1, 3, 224, 224]
#
# - When choosing model FaceDet, the params should be ::
#
#       model_dir = "./FaceDet"
#
#       input_tensor = 'image'
#
#       target_tensor = 'save_infer_model/scale_0'
#
#       x_shape = [1, 3, 240, 320]
#

model_dir = "./ResNet50"
input_tensor = 'inputs'
target_tensor = 'save_infer_model/scale_0.tmp_1'
x_shape = [1, 3, 224, 224]

##################################################################
# **Set the target backend**
# ------------------------------
# Now CINN only supports two backends: X86 and CUDA.
#
# - For CUDA backends, set ``target = DefaultNVGPUTarget()``
#
# - For X86 backends, set ``target = DefaultHostTarget()``
#
if os.path.exists("is_cuda"):
    target = DefaultNVGPUTarget()
else:
    target = DefaultHostTarget()

##################################################################
# Set the input tensor and init interpreter
# -------------------------------------------
executor = Interpreter([input_tensor], [x_shape])

##################################################################
# **Load Model to CINN**
# -------------------------
# Load the paddle model and transform it into CINN IR.
#
# * :code:`model_dir` is the path where the paddle model is stored.
#
# * :code:`target` is the backend to execute model on.
#
# * :code:`params_combined` implies whether the params of paddle model is stored in one file.
#
# * :code:`model_name` is the name of the model. Entering this will enable optimizations for each specific model.
#
# - The model_name for each model is : ``"resnet50"``, ``"mobilenetv2"``, ``"efficientnet"`` and ``"facedet"``.
#
model_name = "resnet50"
params_combined = True
executor.load_paddle_model(model_dir, target, params_combined, model_name)

##################################################################
# **Get input tensor and set input data**
# -----------------------------------------
# Here we use random data as input. In practical applications,
# please replace it with real data according to your needs.
#
a_t = executor.get_tensor(input_tensor)
x_data = np.random.random(x_shape).astype("float32")
a_t.from_numpy(x_data, target)

##################################################################
# Here we set the output tensor's data to zero before running the model.
out = executor.get_tensor(target_tensor)
out.from_numpy(np.zeros(out.shape(), dtype='float32'), target)

##################################################################
# **Execute Model**
# -------------------------
# Execute the model and get output tensor's data.
# :code:`out` is the data of output tensor we want.

executor.run()
out = out.numpy(target)
print("Execution Done!\nResult shape is:\n", out.shape)
print("Result data is:\n", out)

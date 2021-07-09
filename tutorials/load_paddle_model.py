"""
Load and Execute Paddle Model
=====================

In this tutorial, we will show you how to load and execute a paddle model in CINN.
"""

import cinn
from cinn import *
from cinn.frontend import *
from cinn.framework import *
from cinn.common import *
import numpy as np

##################################################################
# Prepare to Load Model
# -------------------------
# Declare the params and prepare to load and execute the paddle model.
#
# - :code:`enable_gpu` implies whether to run CINN on CUDA backends.
#
# - :code:`mnodel_dir` is the path where the paddle model is stored.
#
# - :code:`input_tensor` is the name of input tensor in the model.
#
# - :code:`target_tensor` is the name of output tensor we want.
#
# - :code:`x_shape` is the input tensor's shape of the model

model_dir = "./ResNet18"
input_tensor = 'image'
target_tensor = 'save_infer_model/scale_0'
x_shape = [1, 3, 224, 224]

##################################################################
# Set the target backend

target = DefaultHostTarget()

##################################################################
# Set the input tensor and init interpreter
executor = Interpreter([input_tensor], [x_shape])

##################################################################
# Load Model to CINN
# -------------------------
# Load the paddle model and transform it into CINN IR
#
# * :code:`mnodel_dir` is the path where the paddle model is stored.
#
# * :code:`target` is the backend to execute model on.
#
# * :code:`params_combined` implies whether the params of paddle
# model is stored in one file.

params_combined = True
executor.load_paddle_model(model_dir, target, params_combined)

##################################################################
# Get input tensor and set input data
a_t = executor.get_tensor(input_tensor)
x_data = np.random.random(x_shape).astype("float32")
a_t.from_numpy(x_data, target)

##################################################################
# Get output tensor and init its data to zero.
out = executor.get_tensor(target_tensor)
out.from_numpy(np.zeros(out.shape(), dtype='float32'), target)

##################################################################
# Execute Model
# -------------------------
# Execute the model and get output tensor's data.
# * :code:`out` is the data of output tensor we want.

executor.run()
out = out.numpy(target)
print("Execution Done!\nResult shape is:\n", out.shape)
print("Result data is:\n", out)

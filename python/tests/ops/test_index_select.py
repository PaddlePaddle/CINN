# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import numpy as np
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *

# target = DefaultHostTarget()
target = DefaultNVGPUTarget()

builder = CinnBuilder("test_index_select")
x = builder.create_input(Float(32), (2, 4, 3), "X")
index = builder.create_input(Float(32), (4, ), "Index")
print(f"x: {x.shape()}")
print(f"index: {index.shape()}")

res = builder.index_select(x, index, axis=1)
print(f"res: {repr(res)}")

computation = Computation.build_and_compile(target, builder)

x_data = np.array([[[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3]],
                   [[5.1, 5.2, 5.3], [6.1, 6.2, 6.3], [7.1, 7.2, 7.3],
                    [8.1, 8.2, 8.3]]]).astype("float32")
index_data = np.array([0, 1, 2, 3]).astype("float32")

computation.get_tensor("X").from_numpy(x_data, target)
computation.get_tensor("Index").from_numpy(index_data, target)
computation.execute()

res_tensor = computation.get_tensor(str(res))
res_val = res_tensor.numpy(target)
print(res_val)

print("--------------------------------")
x_tensor = computation.get_tensor(str(x))
x_val = x_tensor.numpy(target)
print(x_val)

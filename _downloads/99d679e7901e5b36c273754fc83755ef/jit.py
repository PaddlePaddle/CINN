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
JIT in CINN
=====================

In this tutorial, we will introduce the JIT module that execute the DSL on X86 and NV GPU.
"""

import cinn
import numpy as np
from cinn import runtime
# sphinx_gallery_thumbnail_path = './paddlepaddle.png'

##################################################################
# declare some variables for latter use
m = cinn.Expr(64)
n = cinn.Expr(64)
k = cinn.Expr(8)
bn = cinn.Expr(32)

##################################################################
# Decleare the computation
# -------------------------
A = cinn.Placeholder("float32", "A", [m, k])
B = cinn.Placeholder("float32", "B", [k, n])

kr = cinn.Var(k.as_int32(), "kr")
C = cinn.compute([
    m, n
], lambda v: cinn.reduce_sum(A(v[0], kr.expr()) * B(kr.expr(), v[1]), [kr]),
                 "C")

stages = cinn.create_stages([C])

target = cinn.Target()
builder = cinn.Module.Builder("matmul", target)

func = cinn.lower("matmul", stages, [A.to_tensor(), B.to_tensor(), C])
builder.add_function(func)
module = builder.build()

##################################################################
# Create a JIT engine.
# ---------------------
#
jit = cinn.ExecutionEngine()
jit.link(module)

##################################################################
# Execute the compiled function
#
a = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device)
b = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device)
c = runtime.cinn_buffer_t(
    np.zeros([m.int(), n.int()]).astype("float32"), runtime.cinn_x86_device)

args = [runtime.cinn_pod_value_t(_) for _ in [a, b, c]]
matmul = jit.lookup("matmul")
matmul(args)

print(c.numpy())

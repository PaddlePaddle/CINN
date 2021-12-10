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
Ways to optimize Matrix Multiplication on CPU
=========================================================

In this tutorial, we will introduce several ways to optimize the performance of the matrix multiplication on X86 CPU.
"""

import cinn
import numpy as np
import time
from cinn import runtime
# sphinx_gallery_thumbnail_path = './paddlepaddle.png'

##################################################################
# Declare the basic computation for a matmul
m = cinn.Expr(1024)
n = cinn.Expr(1024)
k = cinn.Expr(1024)

A = cinn.Placeholder("float32", "A", [m, k])
B = cinn.Placeholder("float32", "B", [k, n])

# k1 is a reduce axis
k1 = cinn.Var(k.as_int32(), "k1")

C = cinn.compute([
    m, n
], lambda vs: cinn.reduce_sum(A(vs[0], k1.expr()) * B(k1.expr(), vs[1]), [k1]),
                 "C")

stages = cinn.create_stages([C])

##################################################################
# Fake input data, here we create a runtime buffer for each input of the generated function.
a = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device, 32)
b = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device, 32)
c = runtime.cinn_buffer_t(
    np.zeros([m.int(), n.int()]).astype("float32"), runtime.cinn_x86_device,
    32)


##################################################################
# Here is a helper function to JIT compile the generated program and test the performance
def test_performance(stages,
                     fn_inputs=[A.to_tensor(), B.to_tensor(), C],
                     input_args=[a, b, c]):
    '''
    fake input data, compile and test program's performance
    '''
    target = cinn.Target()
    builder = cinn.Module.Builder("matmul", target)

    func = cinn.lower("matmul", stages, fn_inputs)
    builder.add_function(func)
    module = builder.build()

    jit = cinn.ExecutionEngine()
    jit.link(module)

    args = [runtime.cinn_pod_value_t(_) for _ in input_args]
    matmul = jit.lookup("matmul")

    repeat = 4

    tic = time.perf_counter()
    for i in range(repeat):
        matmul(args)
    toc = time.perf_counter()
    miniseconds = (toc - tic) / repeat * 1e3
    print(f"Takes {miniseconds:0.3f} ms")


# The basic computation without any schedule has a performance as follows
test_performance(stages)

##################################################################
# Blocking
# ---------------------
stages = cinn.create_stages([C])
bn = 32
i_outer, i_inner, j_outer, j_inner = stages[C].tile(0, 1, bn, bn)
k_outer, k_inner = stages[C].split("k1", 4)
stages[C].reorder([i_outer, j_outer, k_outer, k_inner, i_inner, j_inner])

# The performance is
test_performance(stages)

##################################################################
# Vectorization
# ---------------------
stages = cinn.create_stages([C])
bn = 32
i_outer, i_inner, j_outer, j_inner = stages[C].tile(0, 1, bn, bn)
k_outer, k_inner = stages[C].split("k1", 4)
stages[C].reorder([i_outer, j_outer, k_outer, k_inner, i_inner, j_inner])
stages[C].vectorize(j_inner, 8)

# The performance is
test_performance(stages)

##################################################################
# Loop Permutation
# ---------------------
stages = cinn.create_stages([C])
i_outer, i_inner, j_outer, j_inner = stages[C].tile(0, 1, bn, bn)
k_outer, k_inner = stages[C].split("k1", 4)
stages[C].reorder([i_outer, j_outer, k_outer, i_inner, k_inner, j_inner])
stages[C].vectorize(j_inner, 8)
stages[C].unroll(5)

test_performance(stages)

##################################################################
# Array Packing
# ---------------------
packedB = cinn.compute(
    [n / bn, k, cinn.Expr(bn)], lambda x: B(x[1], x[0] * bn + x[2]), "packedB")
C = cinn.compute([m, n], lambda x: cinn.reduce_sum(
    A(x[0], k1.expr()) * packedB(x[1] / bn, k1.expr(), x[1] % bn), [k1]), "C")

stages = cinn.create_stages([C])
stages[packedB].vectorize(2, 8)

i_outer, i_inner, j_outer, j_inner = stages[C].tile(0, 1, bn, bn)
k_outer, k_inner = stages[C].split("k1", 4)
stages[C].reorder([i_outer, j_outer, k_outer, i_inner, k_inner, j_inner])
stages[C].vectorize(j_inner, 8)

# We make the packedB as another input of the generated function and allocate a runtime buffer for it.
packedB_buf = runtime.cinn_buffer_t(
    np.zeros([n.int() // bn, k.int(), bn]).astype("float32"),
    runtime.cinn_x86_device, 32)

# The final performance is
test_performance(
    stages,
    fn_inputs=[A.to_tensor(), B.to_tensor(), C, packedB],
    input_args=[a, b, c, packedB_buf])

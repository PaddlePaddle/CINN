"""
Schedule Primitives in CINN
===============================

In this tutorial, we will guide you through the examples of using schedule primitives.
"""

import cinn
import numpy as np

##################################################################
# declare some variables for latter use
# Expr is short for expression.
m = cinn.Expr(32)
n = cinn.Expr(8)

print(m, n)
# get the integer contained in an integer expression
print(m.int())

##################################################################
#
# A schedule can be created from a list of Tensors.

# declare an elementwise multiply
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

# create the stages for further schedule
stages = cinn.create_stages([C])

# lower will transform the computation to real code
fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C])
print(fn)

##################################################################
#
# One schedule is composed by multiple stages. We provide several
# methods to schedule each stage.

##################################################################
#
# split
# ------
# :code:`split` can partition a specific axis into two axises by :code: `factor`.
A = cinn.Placeholder('float32', 'A', (m, ))
B = cinn.compute((m, ), lambda v: A(v[0]) * 2., name='B')

stages = cinn.create_stages([B])
i0, i1 = stages[B].split(level=0, factor=4)
fn = cinn.lower("fn", stages, [A.to_tensor(), B])
print(fn)

##################################################################
#
# fuse
# ------
# :code:`fuse` can fuse two specific axises into a axis.
# It is the reverse operation of `split`.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.compute((m, n), lambda v: A(v[0], v[1]) * 2., name='B')

stages = cinn.create_stages([B])
i0 = stages[B].fuse(level0=0, level1=1)
fn = cinn.lower("fn", stages, [A.to_tensor(), B])
print(fn)

##################################################################
#
# tile
# ------
# :code:`tile` can partition two adjacent axises into blocks.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

stages = cinn.create_stages([C])

i, j = stages[C].axis(0), stages[C].axis(1)
i_outer, i_inner, j_inner, j_outer = stages[C].tile(i, j, 4, 4)
fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C])
print(fn)

##################################################################
#
# reorder
# ---------
# :code:`reorder` can reorder the axises in the specified order.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

stages = cinn.create_stages([C])
i0, i1 = stages[C].axis(0), stages[C].axis(1)
stages[C].reorder([i1, i0])

fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C])
print(fn)

##################################################################
#
# unroll
# ------
# :code:`unroll` unroll a specific axis.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

stages = cinn.create_stages([C])
i1 = stages[C].axis(1)
stages[C].unroll(i1)

fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C])
print(fn)

##################################################################
#
# compute_inline
# ----------------
# :code:`compute_inline` marks a stage as inline, then the computation
# body will be expanded and inserted at the location where the tensor
# is referenced.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

# C1[i,j] = C[i,j] + B[i,j]
C1 = cinn.compute([m, n], lambda v: C(v[0], v[1]) + B(v[0], v[1]), "C1")
# C2[i,j] = C1[i,j] + B[i,j]
C2 = cinn.compute([m, n], lambda v: C1(v[0], v[1]) + B(v[0], v[1]), "C2")

stages = cinn.create_stages([C, C1, C2])

stages[C].compute_inline()
stages[C1].compute_inline()

fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C2])
print(fn)

##################################################################
#
# bind
# ----------------
# :code:`bind` can bind a specified axis with a thread axis.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.Placeholder('float32', 'B', (m, n))
C = cinn.compute((m, n), lambda v: A(v[0], v[1]) * B(v[0], v[1]), name='C')

stages = cinn.create_stages([C])
stages[C].bind(0, "blockIdx.x")
stages[C].bind(1, "threadIdx.x")

fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C])
print(fn)

##################################################################
#
# compute_at
# ----------------
# :code:`compute_at` can specify the stage to be computed at
# another stage's scope.
# The input param `other` specifies the other stage.
# The input param `level` specifies the stage's scope(which loop)
# to be computed at.
A = cinn.Placeholder('float32', 'A', (m, n, n))
B = cinn.Placeholder('float32', 'B', (m, n, n))
C = cinn.compute(
    (m, n), lambda v: A(v[0], v[1], v[1]) * B(v[0], v[1], v[1]), name='C')
D = cinn.compute((m, n), lambda v: C(v[0], v[1]) + 1., name='D')
stages = cinn.create_stages([C, D])

print("---------Before compute_at---------")
fn = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C, D])
print(fn)

print("---------After compute_at---------")
stages[C].compute_at(other=stages[D], level=1)
fn2 = cinn.lower("fn", stages, [A.to_tensor(), B.to_tensor(), C, D])
print(fn2)

##################################################################
#
# cache_read
# ------
# :code:`cache_read` can create a cache Tensor and load the origin
# Tensor's data into this buffer.
# It will replace all the reading in the readers with the cache.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.compute((m, n), lambda v: A(v[0], v[1]) * 2., name='B')

stages = cinn.create_stages([B])
ACR = stages[A.to_tensor()].cache_read("local", [B], stages)
fn = cinn.lower("fn", stages, [A.to_tensor(), ACR, B])
print(fn)

##################################################################
#
# cache_write
# ------
# :code:`cache_write` can create a cache for writing to the
# original tensor.
# It will store the data in the cache memory first, then
# write to the output tensor.
A = cinn.Placeholder('float32', 'A', (m, n))
B = cinn.compute((m, n), lambda v: A(v[0], v[1]) * 2., name='B')

stages = cinn.create_stages([B])
BCR = stages[B].cache_write("local", stages)
fn = cinn.lower("fn", stages, [A.to_tensor(), B, BCR])
print(fn)

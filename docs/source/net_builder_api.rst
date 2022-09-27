.. role:: math(raw)
   :format: html latex
..

.. role:: raw-latex(raw)
   :format: latex
..

.. contents::
   :depth: 3
..

NetBuilder Primitive Semantics
===============================

The following describes the semantics of operations defined in the
NetBuilder interface.

1. Scalar Primitive APIs
------------------------

Constant
~~~~~~~~~~~

``Constant(value, name)``\ Create a constant value.

``Args:``

-  value(int\|float\|bool\|list): constant value
-  name(str): the name of the constant

``Returns:`` a scalar constant

2. Unary Primitive APIs
-----------------------

Abs
~~~

``Abs(Variable)`` Element-wise absolute ``x -> abs(x)``

Acos
~~~~

``Acos(Variable)`` Element-wise arc-cosine ``x -> acos(x)``

Acosh
~~~~~

``Acosh(Variable)`` Element-wise hyperbolic arc-cosine ``x -> acosh(x)``

Asin
~~~~

``Asin(Variable)`` Element-wise arc-sine ``x -> asin(x)``

Asinh
~~~~~

``Asinh(Variable)`` Element-wise hyperbolic arc-sine ``x -> asinh(x)``

Atan
~~~~

``Atan(Variable)`` Element-wise arc-tangent ``x -> atan(x)``

Atanh
~~~~~

``Atanh(Variable)`` Element-wise hyperbolic arc-tangent
``x -> atanh(x)``

BitwiseNot
~~~~~~~~~~

``BitwiseNot(Variable)`` Given a input ``x``, return its element-wise
logical not. The mathematical expression is:

-  BitwiseNot(x) = ~ x

Ceil
~~~~

``Ceil(Variable)`` Element-wise ceil ``x -> ⌈x⌉``

Cos
~~~

``Cos(Variable)`` Element-wise cosine ``x -> cos(x)``

Cosh
~~~~

``Cosh(Variable)`` Element-wise hyperbolic cosine ``x -> cosh(x)``

Erf
~~~

``Erf(Variable)`` Given input ``variable``, return its Gauss error
function. The mathematical expression is:

-  .. math:: Erf(x) = \frac{2}{\sqrt[]{\pi}}\int_{0}^{x}e^{-t^2}\text{d}t

Exp
~~~

``Exp(Variable)`` Given input ``variable``, return its exponential
function. The mathematical expression is:

-  .. math:: Exp(x) = e^x

Floor
~~~~~

``Floor(Variable)`` Given input ``variable``, return the greatest
integer less than or equal to ``variable``. The mathematical expression
is:

-  .. math:: Floor(x) = max\{m\in Z \  | \ m \le x \}

Identity
~~~~~~~~

``Identity(Variable)`` Given input ``variable``, return an identical
variable. The mathematical expression is:

-  .. math:: Identity(x) = x

IsFinite
~~~~~~~~

``IsFinite(Variable)`` Element-wise determines if the given input
``variable`` has finite value.

IsInf
~~~~~

``IsInf(Variable)`` Element-wise determines if the given input
``variable`` has infinite value.

IsNan
~~~~~

``IsNan(Variable)`` Element-wise determines if the given input
``variable`` has not-a-number (NaN) value.

Log
~~~

``Log(Variable)`` Given a input ``x``, return its natural logarithm. The
mathematical expression is:

-  .. math:: Log(x) = \log_ex

Log10
~~~~~

``Log10(Variable)`` Given a input ``x``, return its common logarithm.
The mathematical expression is:

-  .. math:: Log10(x) = \log_{10}x

Log2
~~~~

``Log2(Variable)`` Given a input ``x``, return its binary logarithm. The
mathematical expression is:

-  .. math:: Log2(x) = \log_2x

LogicalNot
~~~~~~~~~~

``LogicalNot(Variable)`` Given a input ``x``, return its element-wise
logical not. The mathematical expression is:

-  LogicalNot(x) = ! x

Negative
~~~~~~~~

``Negative(Variable)`` Given an input ``x``, return its negative. The
mathematical expression is:

-  .. math:: Negative(x) = -x

Round
~~~~~

``Round(Variable)`` Given an input ``x``, return the nearest value to
``x`` with halfway cases away from zero. The mathematical expression
is（where :math:`sgn(x)` refers to the sign of ``x``):

-  .. math:: Round(x) = sgn(x)+ \lfloor \left| x \right| + 0.5 \rfloor

Rsqrt
~~~~~

``Rsqrt(Variable)`` Given an input ``x``, return the reciprocal of the
square root of ``x``. The mathematical expression is:

-  .. math:: Rsqrt(x) = \frac{1}{\sqrt{x}}

Sign
~~~~

``Sign(Variable)``, Given an input ``x``, extracts the sign of ``x``,
usually called the signum function. The mathematical expression is：

-  .. math:: Sign(x) = \{ \begin{array}{rc1} -1 & if & x < 0 \\ 0 & if & x = 0 \\ 1 & if & x > 0 \end{array}

Sin
~~~

``Sin(Variable)``, Given an input ``x``, return the sine of ``x``. The
mathematical expression is：

-  .. math:: Sin(x) = \sin{x}

Sinh
~~~~

``Sin(Variable)``, Given an input ``x``, return the hyperbolic sine of
``x``. The mathematical expression is：

-  .. math:: Sinh(x) = \sinh{x}

Sqrt
~~~~

``Sqrt(Variable)`` Given an input variable, return the result of its
square root. The mathematical expression is:

-  .. math:: Sqrt(x) = \sqrt{x}

Tan
~~~

``Tan(Variable)``, Given an input ``x``, return the tangent of ``x``.
The mathematical expression is：

-  .. math:: Tan(x) = \tan{x}

Tanh
~~~~

``Sin(Variable)``, Given an input ``x``, return the hyperbolic tangent
of ``x``. The mathematical expression is：

-  .. math:: Tanh(x) = \tanh{x}

Trunc
~~~~~

``Trunc(Variable)``, Given an input ``x``, return the nearest integer
not greater in magnitude than ``x`` with cutting away (truncates) the
decimal places. The mathematical expression is：

-  .. math:: Trunc(x) = \{ \begin{array}{rc1} \lceil x \rceil & if & x < 0 \\ 0 & if & x = 0 \\ \lfloor x \rfloor & if & x > 0 \end{array}

3. Binary Primitive APIs
------------------------

Add
~~~

``Add(Variable, Variable)`` Given two input variables, return the result
of their element-wise addition. The mathematical expression is:

-  .. math:: Add(x, y) = x + y

BitwiseAnd
~~~~~~~~~~

``BitwiseAnd(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their bitwise logical and. The mathematical expression is:

-  BitwiseAnd(x, y) = x & y

BitwiseOr
~~~~~~~~~

``BitwiseOr(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their bit-wise logical or . The mathematical expression is:

-  BitwiseOr(x, y) = x \| y

BitwiseXor
~~~~~~~~~~

``BitwiseXor(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their bit-wise logical xor. The mathematical expression is:

-  BitwiseXor(x, y) = x ^ y

Div
~~~

``Div(Variable, Variable)`` Given two input variables, return the result
of their element-wise division. The mathematical expression is:

-  .. math:: Div(x, y) = x / y

Dot
~~~

``Dot(Variable, Variable)`` Given two input variables, return the result
of their dot product. The mathematical expression is:

-  .. math:: Dot(x, y) = x^{T}y

FloorDiv
~~~~~~~~

``FloorDiv(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return the greatest integer less than or equal to ``x / y``. The
mathematical expression is:

-  .. math:: FloorDiv(x, y) = max\{m\in Z \  | \ m \le \frac{x}{y} \}

FloorMod
~~~~~~~~

``FloorMod(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return the modulo operation. The mathematical expression is:

-  .. math:: FloorMod(x, y) = x-y*floor(\frac{x}{y})

.. math::

   FloorMod(x, y) = x-y*floor(\frac{x}{y})

.. math:: (a + b)^2 = a^2 + 2ab + b^2

- .. math:: RightShift(x, y) = x >> y

- .. math:: 
   
   RightShift(x, y) = x >> y

.. math::

   y = \textrm{sigmoid}(X\beta - \textrm{offset}) + \epsilon =
   \frac{1}{1 + \textrm{exp}(- X\beta + \textrm{offset})} + \epsilon

LeftShift
~~~~~~~~~

``LeftShift(Variable, Integer)`` Given two inputs ``x`` and ``y``, move
all the bits of ``x`` to left by ``y``. The operation is:

-  .. math:: LeftShift(x, y) = x << y

LogicalAnd
~~~~~~~~~~

``LogicalAnd(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their element-wise logical and. The mathematical expression is:

-  LogicalAnd(x, y) = x && y

LogicalOr
~~~~~~~~~

``LogicalOr(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their element-wise logical or.

- LogicalOr(x, y) = x \|\| y

LogicalXor
~~~~~~~~~~

``LogicalXor(Variable, Variable)`` Given two inputs ``x`` and ``y``,
return their element-wise logical xor.

- LogicalXor(x, y) = (x \|\| y) && !(x && y)

Max
~~~

``Max(Variable, Variable)`` Given two inputs ``x`` and ``y``, return the
maximum one.

Min
~~~

``Min(Variable, Variable)`` Given two inputs ``x`` and ``y``, return the
minimum one.

Mod
~~~

``Mod(Variable, Variable)`` Given two inputs ``x`` and ``y``, return
their mod value. The mathematical expression is:

-  .. math:: Mod(x, y) = x % y

Mul
~~~

``Mul(Variable, Variable)`` Given two input variables, return the result
of their element-wise multiplication. The mathematical expression is:

-  .. math:: Mul(x, y) = x * y

Power
~~~~~

``Mod(Variable, Variable)`` Given two inputs ``x`` and ``y``
sequentially, where ``x`` is called the base and ``y`` is the exponent,
this operator returns the product of multiplying ``y`` on base ``x``.
The mathematical expression is: 

-  .. math:: Power(x, y) = x ^ {y}

RightShift
~~~~~~~~~~

``RightShift(Variable, Variable)`` Given two inputs ``x`` and ``y``,
move all the bits of ``x`` to right by ``y``, if ``x`` is a signed type
then performs an arithmetic shift otherwise a logical shift. The
mathematical expression is:

-  .. math:: RightShift(x, y) = x >> y

Sub
~~~

``Sub(Variable, Variable)`` Given two input variables, return the result
of their element-wise subtraction. The mathematical expression is:

-  .. math:: Sub(x, y) = x - y

4. Complex Primitive APIs
-------------------------

BnGradBiasScale
~~~~~~~~~~~~~~~

``BnGradBiasScale(input, grad, save_mean)`` Compute the gradient of bias
and scale in batch normalization.

``Args:``

-  input: input tensor of batch normalization.
-  grad: gradient of output tensor of batch normalization.
-  save\_mean: the mean of input tensor which was saved when doing batch
   normalization forward computation.

``Returns:`` Two gradient tensors of bias and scale.

BnMeanVariance
~~~~~~~~~~~~~~

``BnMeanVariance(input)`` Compute the sum of input and input^2 in batch
normalization.

``Args:``

-  input: input tensor of batch normalization.

``Returns:`` Two tensors for the sum of input and input^2.

BroadcastTo
~~~~~~~~~~~

``BroadcastTo(var, out_shape, broadcast_axes)`` Broadcast the input
tensor to the target shape by duplicating the elements according to the
broadcast\_axes.

``Args:``

-  var: Input tensor to broadcast
-  out\_shape(list[int]\|tuple[int]): The sizes of the dimensions of the
   target shape.
-  broadcast\_axes(list[int]\|tuple[int]): The target axis in the target
   shape which the input shape's ith axis mapped to. Axis starts from 0.

``Returns:`` a tensor after expanding size and rank

``Examples:``

::

    Variable out_var = net_builder. BroadcastTo(input_var, {1, 64, 112, 112}, {1});

If input\_var[64] broadcasts to out\_var[1, 64, 112, 112], then
out\_shape is [1, 64, 112, 112] and broadcast\_axes are [1]. The i-th
axis of the input shape is mapped to the broadcast\_axes[i]-th axis of
the output shape. Notes that the i-th axis's dimension of the input must
be 1 or the same as the broadcast\_axes[i]-th axis dimension of the
output shape. And the sizes of the input shape should be the same as the
size of the broadcast\_axes which indicates the mapping relation. In
this case, the 0‘th axis of the input shape is mapped to the 1’th axis
of the output shape. And both dimensions are 64. The left
degenerate-axes then broadcast along these degenerate axes to reach the
output shape.

Compare
~~~~~~~

``Compare(Variable, Variable, ComparisonKind)`` Given two input
variables, return the result of their element-wise comparison. The value
of ``ComparisonKind`` can be ``kEq``, ``kNe``, ``kGe``, ``kGt``, ``kLe``
or ``kLt``. Its semantics can be expressed by the following formula:

.. math:: 

   Compare(x, y, kind) = 
   \left\{  
             \begin{array}{**lr**}  
				x == y & if & kind =  kEq \\
				x \neq y & if & kind =  kNe \\
				x \ge y & if & kind =  kGe \\
				x \gt y & if & kind =  kGt \\
				x \le y & if & kind =  kLe \\
				x \lt y & if & kind =  kLt
             \end{array}  
   \right.

Concat
~~~~~~

``Concat(input_vars, axis=0)``\ Concatenate the input tensors along an
existing axis.

``Args:``

-  input\_vars(list\|tuple): input tensors to concatenate
-  axis(int,optional): Specify the axis to concatenate the input
   tensors. Default is 0.

``Returns:`` a tensor after concatenation

Conv
~~~~

``Conv(lhs, rhs, strides, paddings, dilations, groups, conv_type, data_format, padding_algorithm, output_shape)``
Convolution operation with input tensor lhs and rhs.

``Args:``

-  lhs: Input tensor lhs.
-  rhs: Input tensor rhs.
-  strides: A list of 2 integers, specifying the strides of the
   convolution along with the height and width. Specifying any stride
   value != 1 is incompatible with specifying any dilation\_rate value
   != 1.
-  paddings: A list of 2 integers. It has the form [pad\_vertical,
   pad\_horizontal].
-  dilations: A list of 2 integers, specifying the dilation rate to use
   for dilated convolution. Currently, specifying any dilation\_rate
   value != 1 is incompatible with specifying any stride value != 1.
-  groups: The group's number of the convolution. According to grouped
   convolution in Alex Krizhevsky’s Deep CNN paper. The default value is
   1.
-  conv\_type: the type of convolution, it should be one of{*'forward',
   'backward\_data', 'backward\_filter'*}.

   -  conv\_type = *forward*. lhs is input tensor, rhs is weight tensor.
   -  conv\_type = *backward\_data*. lhs is weight tensor, rhs is
      gradient tensor.
   -  conv\_type = *backward\_filter*. lhs is input tensor, rhs is
      gradient tensor.

-  data\_format: Data format that specifies the layout of input. It can
   be “NCHW” or “NHWC”. The default value is “NCHW”.
-  padding\_algorithm: The algorithm used for padding. The default value
   is "EXPLICIT".
-  output\_shape: The shape of the output tensor. The default value is
   {}. output\_shape can't be Omitted, when conv\_type =
   *"backward\_data" or "backward\_filter"*, as the shape's inference is
   irreversible.

``Returns:`` A tensor after convolution.

``Examples:``

::

    // convolution forward
    // x = [16,16,28,28], filter = [32, 16, 3, 3]
    // strides = {1, 1}, paddings = {1, 1}, dilations = {1, 1}
    // y = [16, 32, 28, 28]
    Variable y = net_builder.Conv(x, filter, {1,1}, {1,1}, {1,1}, 1, "forward", "NCHW", "EXPLICIT", {});

    // convolution backward_data
    // grad_y = [16, 32, 28, 28], filter = [32, 16, 3, 3]
    // strides = {1, 1}, paddings = {1, 1}, dilations = {1, 1}
    // grad_x = [16, 16, 28, 28]
    Variable grad_x = net_builder.Conv(filter, grad_y, {1, 1}, {1, 1}, {1, 1}, 1, "backward_data", "NCHW", "EXPLICIT", {16, 16, 28, 28});

    // convolution backward_filter
    // grad_y = [16, 32, 28, 28], x = [16, 16, 28 ,28]
    // strides = {1, 1}, paddings = {1, 1}, dilations = {1, 1}
    // grad_filter = [32, 16, 3, 3]
    grad_filte = net_builder.Conv(x, grad_y, {1, 1}, {1, 1}, {1, 1}, 1, "backward_filter", "NCHW", "EXPLICIT", {32, 16, 3, 3});

ReduceSum
~~~~~~

``Reduce(input, kind, dim, keep_dim)`` Reduce on input tensors along the
given dimensions.

``Args:``

-  input: The input tensor.
-  kind: An enumerate value to specify the reduce type, the value should
   be on of {*'kSum', 'kProd', 'kMa', 'kMin'*}.
-  dim: A list of integers, specifying the reduced dimensions. the value
   must be along[0, size(input)).
-  keep\_dim: A boolean value, specifying whether to keep the output
   shape size.

``Returns:`` A tensor after reduce.

::

    // Case 1
    // x = [128, 128, 16, 16]
    // kind = sum, dim = {2, 3}
    // out = [128, 128] when keep_dim = false
    out = net_builder.ReduceSum(x, {2, 3}, false);

    // Case 2
    // x = [128, 128, 16, 16]
    // kind = sum, dim = {2, 3}
    // out = [128, 128, 1, 1] when keep_dim = true
    out = net_builder.ReduceSum(x, {2, 3}, true);

Reshape
~~~~~~~

``Reshape(input, shape)`` Reshape the input tensor to a given shape.

``Args:``

-  input: The input tensor.
-  shape: A list of integers, define the target shape. At most one
   dimension of the target shape can be -1.

``Returns:`` A tensor after reduce.

Reverse
~~~~~~~

``Reverse(input, axis)`` Reverse the elements of the input tensor on a
given axis.

``Args:``

-  input: The input tensor.
-  axis: A list of integers, specifying the axis to be reversed.

``Returns:`` A tensor after reverse.

Select
~~~~~~

``Select(condition, true_value, false_value)`` Select elements from
input tensors *rue\_value* and *false\_value*, based on the values of
condition tensor.

``Args:``

-  condition: Condition tensor for elements selection.
-  true\_value: True value tensor.
-  false\_value: False value tensor.

``Returns:`` A tensor after select.

Slice
~~~~~

``Slice(input, axes, starts, ends)`` Slicing extracts a sub-array from
the input array. The sub-array is of the same rank as the input and
contains the values inside a bounding box within the input array where
the dimensions and indices of the bounding box are given as arguments to
the slice operation.

``Args:``

-  input: The input tensor.
-  axes: A list of integers specifying the dimensions to slice.
-  starts: A list of Integers containing the starting indices of the
   slice for dimension in axes. Values must be greater than or equal to
   zero.
-  ends: A List of integers containing the ending indices of the slice
   for dimension in axes. Values must be greater than starts and less or
   equal to the length of the dimension.

Transpose
~~~~~~~~~

``Transpose(input, axis)`` Permutes the operand dimensions with the
given axis.

``Args:``

-  input: The input tensor.
-  axis: A list of integers for the permutation. The size of the axis
   should be equal to or lesser than the size dimension，and values must
   be along [0, size(input)).

``Returns:`` A tensor after Transpose.

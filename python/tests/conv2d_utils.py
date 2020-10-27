#!/usr/bin/env python3
import paddle as paddle
import paddle.fluid as fluid
import numpy as np


def conv2d_native(inputs_data, input_shape, filter_size, attrs, is_depthwise):
    paddle.enable_static()
    main_program = fluid.Program()
    with fluid.program_guard(main_program, fluid.Program()):
        padding = [0, 0]
        stride = [1, 1]
        dilation = [1, 1]
        data_format = "NCHW"
        groups = 1
        for key in attrs.attr_store:
            if key == "stride":
                stride = attrs.get_attr("stride")
            elif key == "padding":
                padding = attrs.get_attr("padding")
            elif key == "dilation":
                dilation = attrs.get_attr("dilation")
            elif key == "groups":
                groups = attrs.get_attr("groups")
            elif key == "data_format":
                data_format = attrs.get_attr("data_format")
            else:
                raise ValueError("attr_store {} is not supported".format(key))

        img = fluid.layers.data(
            name='img', shape=input_shape[1:], dtype='float32')
        if is_depthwise:
            if data_format == "NCHW":
                cin_index = 1
            else:
                cin_index = 3
            filter_size_new = [
                filter_size[1] * input_shape[cin_index],
                filter_size[0] // groups, filter_size[2], filter_size[3]
            ]
        else:
            filter_size_new = filter_size
        param = fluid.initializer.NumpyArrayInitializer(
            np.array(
                inputs_data[1]).reshape(filter_size_new).astype("float32"))
        # filter: (c_out, c_in // group, kernel_h, kernel_w)
        filter_hw = list(filter_size_new[2:4])
        if data_format == "NHWC":
            filter_hw = list(filter_size_new[1:3])
        if isinstance(stride, int):
            stride = [stride.copy(), stride.copy()]
        if isinstance(padding, int):
            padding = [padding.copy(), padding.copy()]
        if isinstance(dilation, int):
            dilation = [dilation.copy(), dilation.copy()]

        res = fluid.layers.conv2d(
            input=img,
            num_filters=filter_size_new[0],
            filter_size=filter_hw,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            param_attr=param,
            data_format=data_format)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(inputs_data[0]).reshape(input_shape).astype("float32")
        output = exe.run(feed={"img": x}, fetch_list=[res])
        output = np.array(output)
        print("output's shape is:", output.shape)

    res_shape = output.shape[1:]

    if is_depthwise:
        return output, [res_shape]
    else:
        return output, [res_shape]

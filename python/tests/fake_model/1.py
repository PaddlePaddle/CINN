from paddle import fluid
import numpy as np

exe = fluid.Executor(fluid.CPUPlace())

path = './model2'

[inference_program, feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(
     dirname=path, executor=exe)

data = np.ones([10, 10], dtype='float32')

results = exe.run(
    inference_program,
    feed={feed_target_names[0]: data},
    fetch_list=fetch_targets)

result = results[0]
print(result)

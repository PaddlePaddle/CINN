import argparse
import time
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def main():
    args = parse_args()

    config = set_config(args)

    predictor = create_paddle_predictor(config)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    fake_input = np.random.randn(512, 512).astype("float32")
    input_tensor.reshape([512, 512])
    input_tensor.copy_from_cpu(fake_input)

    input_tensor2 = predictor.get_input_tensor(input_names[1])
    fake_input2 = np.random.randn(512, 512).astype("float32")
    input_tensor2.reshape([512, 512])
    input_tensor2.copy_from_cpu(fake_input2)

    for _ in range(0, 10):
        predictor.zero_copy_run()
    #time_start = time.time()
    for i in range(0, 500):
        predictor.zero_copy_run()
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu()
    #time_end = time.time()
    #print('totally cost',(time_end-time_start)/2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="model filename")
    #parser.add_argument("--params_file", type=str, help="parameter filename")
    #parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    return parser.parse_args()


def set_config(args):
    config = AnalysisConfig(args.model_dir)
    config.enable_profile()
    config.enable_use_gpu(1000, 1)
    config.gpu_device_id()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.switch_ir_optim(False)
    #To test cpu backend, just uncomment the following 2 lines.
    #config.disable_gpu()
    #config.enable_mkldnn()
    return config


if __name__ == "__main__":
    main()

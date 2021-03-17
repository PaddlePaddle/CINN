#!/usr/bin/env python3
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *
import numpy as np
import sys

assert len(sys.argv) == 2
enable_gpu = sys.argv.pop()


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def atest_conv2d(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 7, 7])
        b = Variable("E").set_type(Float(32)).set_shape([512, 512, 3, 3])
        c = prog.conv2d(a, b, {
            "stride": [1, 1],
            "dilation": [1, 1],
            "padding": [1, 1]
        })
        tensor_data = [
            np.random.random([2, 512, 7, 7]).astype("float32"),
            np.random.random([512, 512, 3, 3]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 2000,
            "TESTING [conv2d] time cost with shape [2,512,7,7]...")

    def atest_softmax(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([1024, 2048])
        c = prog.softmax(a, {})
        tensor_data = [np.random.random([1024, 2048]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [softmax] time cost with shape [1024,2048]...")

    def atest_matmul(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([512, 512])
        b = Variable("B").set_type(Float(32)).set_shape([512, 512])
        c = prog.mul(a, b, 1, 1)
        tensor_data = [
            np.random.random([512, 512]).astype("float32"),
            np.random.random([512, 512]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [matmul] time cost with shape [512,512]...")

    def atest_matmul1(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([128, 512])
        b = Variable("B").set_type(Float(32)).set_shape([256, 512])
        c = Variable("C").set_type(Float(32)).set_shape([128, 256])
        d = prog.mulbias(a, b, c, 1, 1)
        tensor_data = [
            np.random.random([128, 512]).astype("float32"),
            np.random.random([256, 512]).astype("float32"),
            np.random.random([128, 256]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b, c], tensor_data, d, 200,
            "TESTING [mulbias] time cost with shape [128,512]*[256,512]...")

    def atest_matmul2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([128, 512])
        b = Variable("B").set_type(Float(32)).set_shape([256, 512])
        c = Variable("C").set_type(Float(32)).set_shape([128, 256])
        d = prog.mul(a, b, 1, 1)
        e = prog.add(d, c)
        tensor_data = [
            np.random.random([128, 512]).astype("float32"),
            np.random.random([256, 512]).astype("float32"),
            np.random.random([128, 256]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b, c], tensor_data, e, 200,
            "TESTING [mul and add] time cost with shape [128,512]*[256,512]..."
        )

    def atest_matmul(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([512, 512])
        b = Variable("B").set_type(Float(32)).set_shape([512, 512])
        c = Variable("C").set_type(Float(32)).set_shape([512, 512])
        d = prog.mul(a, b, 1, 1)
        # e = prog.add(d, c)
        tensor_data = [
            np.random.random([512, 512]).astype("float32"),
            np.random.random([512, 512]).astype("float32")
        ]
        result = prog.test_benchmark_with_code(
            self.target, [a, b], tensor_data, d, 200,
            "TESTING [matmul] time cost with shape [512,512]...", '''
            extern "C" {
#include "cinn_cuda_runtime_source.cuh"
#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif

 __global__
 void fn_mul_0_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Mul_output)
 {
   const float* A_reshape = A;
   const float* B_reshape = B;
   float* Mul_output__reduce_init = Mul_output;
   if ((blockIdx.x < 512)) {
   {
     if ((threadIdx.x < 256)) {
     {
       for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
         Mul_output__reduce_init[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] = 0;
       };
     }
     };
   }
   };
   if ((blockIdx.x < 512)) {
   {
     if ((threadIdx.x < 256)) {
     {
       for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
        for (int32_t axis_k = 0; axis_k < 512; axis_k += 1) {
          Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] = (Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] + (A_reshape[((512 * blockIdx.x) + axis_k)] * B_reshape[((512 * axis_k) + ((2 * threadIdx.x) + j_inner))])) + Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))];
         };
       };
     }
     };
  }
  };
 }
 }''')

    def test_pool2d(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 64, 112, 112])
        c = prog.pool2d(
            a, {
                "kernel_size": (3, 3),
                "stride_size": (2, 2),
                "padding_size": (1, 1, 1, 1),
                "pool_type": "max"
            })
        tensor_data = [np.random.random([2, 64, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 2000,
            "TESTING [pool2d] time cost with shape [2, 64, 112, 112]...")

    def atest_elementwise1(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        b = Variable("B").set_type(Float(32)).set_shape([64, 64])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([64, 64]).astype("float32"),
            np.random.random([64, 64]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [elementwise_add] time cost with shape [64, 64]...")
        result = result.numpy(self.target).reshape(-1)
        self.assertTrue(
            np.allclose(
                (tensor_data[0] + tensor_data[1]).reshape(-1),
                result,
                atol=1e-4))

    def atest_elementwise2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        b = Variable("B").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([2, 512, 112, 112]).astype("float32"),
            np.random.random([2, 512, 112, 112]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [elementwise_add] time cost with shape [2, 512, 112, 112]..."
        )
        result = result.numpy(self.target).reshape(-1)
        self.assertTrue(
            np.allclose(
                (tensor_data[0] + tensor_data[1]).reshape(-1),
                result,
                atol=1e-4))

    def atest_elementwise2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([4, 1024])
        b = Variable("B").set_type(Float(32)).set_shape([4, 1024])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([4, 1024]).astype("float32"),
            np.random.random([4, 1024]).astype("float32")
        ]
        result = prog.test_benchmark_with_code(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [elementwise_add] time cost with input code...",
            '''extern "C" {

__global__
void fn_elementwise_add_0_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ EleAdd_Out_0)
{

      EleAdd_Out_0[1024 * blockIdx.x + threadIdx.x] = (A[1024 * blockIdx.x + threadIdx.x] + B[1024 * blockIdx.x + threadIdx.x]);
}

}''')

    def atest_batchnorm(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 32, 32])
        b = Variable("B").set_type(Float(32)).set_shape([512])
        c = Variable("C").set_type(Float(32)).set_shape([512])
        d = Variable("D").set_type(Float(32)).set_shape([512])
        e = Variable("E").set_type(Float(32)).set_shape([512])
        f = prog.batchnorm(a, b, c, d, e, {})
        tensor_data = [
            np.random.random([2, 512, 32, 32]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b, c, d, e], tensor_data, f, 1000,
            "TESTING [batchnorm] time cost with shape [2, 512, 32, 32]...")

    def atest_batchnorm2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 64, 8, 8])
        b = Variable("B").set_type(Float(32)).set_shape([64])
        c = Variable("C").set_type(Float(32)).set_shape([64])
        d = Variable("D").set_type(Float(32)).set_shape([64])
        e = Variable("E").set_type(Float(32)).set_shape([64])
        f = prog.batchnorm(a, b, c, d, e, {})
        tensor_data = [
            np.random.random([2, 64, 8, 8]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b, c, d, e], tensor_data, f, 200,
            "TESTING [batchnorm] time cost with shape [2, 64, 8, 8]...")

    def atest_relu3(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.relu(a)
        tensor_data = [np.random.random([2, 512, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [relu] time cost with shape [2,512,112,112]...")

    def atest_relu(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        c = prog.sigmoid(a)
        tensor_data = [np.random.random([64, 64]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [sigmoid] time cost with shape [64,64]...")

    def atest_relu2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.sigmoid(a)
        tensor_data = [np.random.random([2, 512, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [sigmoid] time cost with shape [2,512,112,112]...")


if __name__ == "__main__":
    unittest.main()

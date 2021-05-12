#include <gtest/gtest.h>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/utils/timer.h"

namespace cinn {

TEST(test03_conv, schedule) {
  Expr n          = Expr(1);
  Expr c_in       = Expr(3);
  Expr h_in       = Expr(224);
  Expr w_in       = Expr(224);
  Expr c_out      = Expr(64);
  Expr h_f        = Expr(7);
  Expr w_f        = Expr(7);
  Expr dilation_h = Expr(1);
  Expr dilation_w = Expr(1);
  Expr pad_h      = Expr(3);
  Expr pad_w      = Expr(3);
  Expr stride_h   = Expr(2);
  Expr stride_w   = Expr(2);

  Placeholder<float> input("input", {n, c_in, h_in, w_in});
  Placeholder<float> weights("weights", {c_out, c_in, h_f, w_f});
  std::vector<Expr> input_pad_shape = {n, c_in, h_in + 2 * pad_h, w_in + 2 * pad_w};
  int input_pad_h                   = 224 + 2 * 3;
  int input_pad_w                   = 224 + 2 * 3;

  Expr ic_bn    = Expr(3);
  Expr oc_bn    = Expr(16);
  Expr ic_chunk = c_in / ic_bn;
  Expr oc_chunk = c_out / oc_bn;

  std::vector<Expr> output_shape = {
      n,                                                                 // B
      c_out,                                                             // O
      (h_in - ((h_f - 1) * dilation_h + 1) + 2 * pad_h) / stride_h + 1,  // H
      (w_in - ((w_f - 1) * dilation_w + 1) + 2 * pad_w) / stride_w + 1   // W
  };
  int out_h = (224 - 7 + 2 * 3) / 2 + 1;
  int out_w = (224 - 7 + 2 * 3) / 2 + 1;

  // pack_data, 4D->5D
  auto data = Compute(
      {n, ic_chunk, h_in, w_in, ic_bn},
      [=](Expr bs, Expr c, Expr h, Expr w, Expr vc) { return input(bs, c * ic_bn + vc, h, w); },
      UniqName("data_vec"));
  // pack_kernel, 4D->6D
  auto kernel = Compute(
      {oc_chunk, ic_chunk, h_f, w_f, ic_bn, oc_bn},
      [=](Expr occ, Expr icc, Expr h, Expr w, Expr icb, Expr ocb) {
        return weights(occ * oc_bn + ocb, icc * ic_bn + icb, h, w);
      },
      UniqName("kernel_vec"));

  auto input_pad = Compute(
      {n, ic_chunk, h_in + 2 * pad_h, w_in + 2 * pad_w, ic_bn},
      [=](Expr bs, Expr cc, Expr yy, Expr xx, Expr vc) {
        auto cond = lang::logic_and({yy >= pad_h, yy - pad_h < h_in, xx >= pad_w, xx - pad_w < w_in});
        return ir::Select::Make(cond, data(bs, cc, yy - pad_h, xx - pad_w, vc), ir::Zero(input->type()));
      },
      UniqName("input_pad"));

  Var fc(c_in, UniqName("fc"));
  Var fy(h_f, UniqName("fy"));
  Var fx(w_f, UniqName("fx"));

  auto packed_out = Compute(
      {n, oc_chunk, Expr(out_h), Expr(out_w), oc_bn},
      [=](Expr n, Expr oc_chunk, Expr oh, Expr ow, Expr oc_block) {
        return lang::ReduceSum(input_pad(n, fc / ic_bn, oh * stride_h + fy, ow * stride_w + fx, fc % ic_bn) *
                                   kernel(oc_chunk, fc / ic_bn, fy, fx, fc % ic_bn, oc_block),
                               {fc, fy, fx});
      },
      UniqName("conv2d_nchwc"));

  auto res = Compute(
      {n, c_out, Expr(out_h), Expr(out_w)},
      [=](Expr n, Expr c, Expr h, Expr w) { return packed_out(n, c / oc_bn, h, w, c % oc_bn); },
      UniqName("res"));

  Target target(Target::OS::Linux, Target::Arch::X86, Target::Bit::k64);

  Module::Builder builder("conv", target);

  auto stages = CreateStages({data, kernel, input_pad, packed_out, res});
  // schedule
  // data_vec
  stages[data]->Fuse(0, 1);
  stages[data]->Fuse(0, 1);

  // kernel_vec
  // oc_chunk, ic_chunk, oh, ow, ic_block, oc_block -> oc_chunk, oh, ic_chunk, ow, ic_block, oc_block
  stages[kernel]->Reorder({2, 1});
  stages[kernel]->Fuse(0, 1);

  // schedule pad
  int input_pad_dims = stages[input_pad]->n_out_dims();
  stages[input_pad]->Fuse(0, 1);
  stages[input_pad]->Fuse(0, 1);
  stages[input_pad]->Vectorize(2, 3);

  // cache write
  std::vector<ir::Tensor> writers;
  auto CC = stages[packed_out]->CacheWrite2("local", stages, packed_out);

  // C
  // batch, oc_chunk, oh, ow, oc_block
  stages[packed_out]->Split(3, 8);
  stages[packed_out]->Fuse(0, 1);
  stages[packed_out]->Fuse(0, 1);
  stages[packed_out]->Vectorize(3, 16);

  // CC
  stages[CC]->ComputeAt2(stages[packed_out], 1);
  stages[CC]->Split(4, 3);
  stages[CC]->Reorder({4, 6, 7, 5, 2, 3});
  stages[CC]->Vectorize(6, 16);
  // out
  // n, oc, oh, ow
  stages[res]->Split(1, 16);
  stages[res]->Split(4, 8);
  // n, oc_chunk, oc_block, oh, ow_chunk, ow_block -> n, oc_chunk, oh, ow_chunk, ow_block, oc_block
  stages[res]->Reorder({3, 4, 5, 2});

  auto func = Lower("conv_schedule", stages, {input, weights, res});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test03_convolution_schedule1.h").c_source("./test03_convolution_schedule1.cc");
  compiler.Compile(builder.Build(), outputs);

  auto module = builder.Build();
  auto jit    = cinn::backends::SimpleJIT::Create();
  jit->Link(module, true);
  auto conv_fn = reinterpret_cast<void (*)(void**, int32_t)>(jit->Lookup("conv_schedule"));
  cinn::utils::Timer timer;
  const int repeat = 10;

  auto* A = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device,
                                cinn_float32_t(),
                                {n.as_int32(), c_in.as_int32(), h_in.as_int32(), w_in.as_int32()},
                                32);
  auto* B = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device,
                                cinn_float32_t(),
                                {c_out.as_int32(), c_in.as_int32(), h_f.as_int32(), w_f.as_int32()},
                                32);
  auto* C = cinn_buffer_t::new_(
      cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {n.as_int32(), c_out.as_int32(), out_h, out_w}, 32);
  auto* input_pad_buffer  = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device,
                                               cinn_float32_t(),
                                               {n.as_int32(), c_in.as_int32(), input_pad_h, input_pad_w},
                                               32);
  auto* packed_out_buffer = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device,
                                                cinn_float32_t(),
                                                {n.as_int32(), c_out.as_int32() / 32, out_h, out_w, 32},
                                                32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C);
  cinn_pod_value_t A_arg(A);
  cinn_pod_value_t B_arg(B);
  cinn_pod_value_t C_arg(C);
  cinn_pod_value_t input_pad_arg(input_pad_buffer);
  cinn_pod_value_t packed_out_arg(packed_out_buffer);

  cinn_pod_value_t args[] = {A_arg, B_arg, C_arg};

  for (int i = 0; i < repeat; i++) conv_fn(reinterpret_cast<void**>(args), 3);
  timer.Start();
  for (int i = 0; i < repeat; i++) conv_fn(reinterpret_cast<void**>(args), 3);
  LOG(INFO) << timer.Stop() / repeat;
}

}  // namespace cinn

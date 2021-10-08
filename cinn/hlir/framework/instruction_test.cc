#include "cinn/hlir/framework/instruction.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/common/test_helper.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(int M, int N) {
  Expr m(M);
  Expr n(N);

  Placeholder<float> x("x", {m, n});
  Placeholder<float> y("y", {m, n});

  auto z = Compute(
      {m, n}, [=](Expr i, Expr j) { return x(i, j) + y(i, j); }, "z");

  auto stages = CreateStages({z});
  auto fn     = Lower("fn", stages, {x, y, z});

  ir::Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

TEST(Instruction, basic) {
  const int M = 10;
  const int N = 20;

  Scope scope;

  auto get_tensor = [&](const std::string& name) {
    auto* var    = scope.Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);
    return tensor;
  };

  for (auto& name : std::vector<std::string>({"x", "y", "z"})) {
    auto tensor = get_tensor(name);
    tensor->Resize(Shape{{M, N}});
    auto* data = tensor->mutable_data<float>(common::DefaultHostTarget());
    for (int i = 0; i < M * N; i++) {
      data[i] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
  }

  // create Instruction
  Instruction instr(common::DefaultHostTarget(), &scope, {"x", "y"}, {"z"});
  auto jit     = GetLoweredFunc(M, N);
  auto fn_addr = jit->Lookup("fn");
  CHECK(fn_addr);

  instr.SetLoweredFunc(reinterpret_cast<lower_func_ptr_t>(fn_addr));
  instr.Run();

  // check result
  {
    auto xd = get_tensor("x")->data<float>();
    auto yd = get_tensor("y")->data<float>();
    auto zd = get_tensor("z")->data<float>();

    for (int i = 0; i < M * N; i++) {
      LOG_FIRST_N(INFO, 3) << "data: " << xd[i] << " + " << yd[i] << " = " << zd[i];
      ASSERT_NEAR(xd[i] + yd[i], zd[i], 1e-5);
    }
  }
}

#ifdef CINN_WITH_CUDNN

class TestInstruction : public Instruction {
 public:
  TestInstruction(const Target& target,
                  Scope* scope,
                  const std::vector<std::string>& in_args,
                  const std::vector<std::string>& out_args,
                  const std::string& func_name)
      : Instruction(target, scope, in_args, out_args, func_name) {}
  void SetAttr(const std::vector<int>& _attrs) { attrs = _attrs; }
  void SetStrAttr(const std::vector<std::string>& _str_attrs) { str_attrs = _str_attrs; }
  void SetArgs(const std::vector<cinn_pod_value_t>& _args) { pod_args = _args; }

  void RunX() {
    absl::flat_hash_map<std::string, int> attrs_map = {
        {"input_n", attrs[0]},     {"input_c", attrs[1]},     {"input_h", attrs[2]},   {"input_w", attrs[3]},
        {"weights_n", attrs[4]},   {"weights_c", attrs[5]},   {"weights_h", attrs[6]}, {"weights_w", attrs[7]},
        {"pad_h", attrs[8]},       {"pad_w", attrs[9]},       {"stride_h", attrs[10]}, {"stride_w", attrs[11]},
        {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]}, {"groups", attrs[14]},   {"output_n", attrs[15]},
        {"output_c", attrs[16]},   {"output_h", attrs[17]},   {"output_w", attrs[18]},
    };
    if (str_attrs[0] == "forward") {
      // input weight output
      runtime::cuda::cinn_gpu_cudnn_conv2d(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
    } else if (str_attrs[0] == "backward_data") {
      // weight dy dx
      runtime::cuda::cinn_gpu_cudnn_conv2d_backward_data(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
    } else {
      // input dy dx
      runtime::cuda::cinn_gpu_cudnn_conv2d_backward_filter(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
    }
  }

 private:
  std::vector<cinn_pod_value_t> pod_args;
};

TEST(Instruction, CONV_FORWARD) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;

  std::vector<int> attrs = {in, ic, ih, iw, fn, fc, fh, fw, ph, pw, sh, sw, dila_h, dila_w, group, on, oc, oh, ow};
  /*
  absl::flat_hash_map<std::string, int> attrs = {
      {"input_n", in},   {"input_c", ic},   {"input_h", ih},        {"input_w", iw},        {"weights_n", fn},
      {"weights_c", fc}, {"weights_h", fh}, {"weights_w", fw},      {"pad_h", ph},          {"pad_w", pw},
      {"stride_h", sh},  {"stride_w", sw},  {"dilation_h", dila_h}, {"dilation_w", dila_w}, {"groups", group},
      {"output_n", on},  {"output_c", oc},  {"output_h", oh},       {"output_w", ow},
  };
  */

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x = common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w = common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  std::vector<cinn_pod_value_t> args = {x, w, y};
  std::vector<std::string> str_attrs = {"forward"};
  instr.SetAttr(attrs);
  instr.SetStrAttr(str_attrs);
  instr.SetArgs(args);

  instr.RunX();

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

TEST(Instruction, CONV_BACKWARD_DATA) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;

  std::vector<int> attrs = {in, ic, ih, iw, fn, fc, fh, fw, ph, pw, sh, sw, dila_h, dila_w, group, on, oc, oh, ow};
  /*
  absl::flat_hash_map<std::string, int> attrs = {
      {"input_n", in},   {"input_c", ic},   {"input_h", ih},        {"input_w", iw},        {"weights_n", fn},
      {"weights_c", fc}, {"weights_h", fh}, {"weights_w", fw},      {"pad_h", ph},          {"pad_w", pw},
      {"stride_h", sh},  {"stride_w", sw},  {"dilation_h", dila_h}, {"dilation_w", dila_w}, {"groups", group},
      {"output_n", on},  {"output_c", oc},  {"output_h", oh},       {"output_w", ow},
  };
  */

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x = common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w = common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  std::vector<cinn_pod_value_t> args = {w, y, x};
  std::vector<std::string> str_attrs = {"backward_data"};
  instr.SetAttr(attrs);
  instr.SetStrAttr(str_attrs);
  instr.SetArgs(args);

  instr.RunX();

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

TEST(Instruction, CONV_BACKWARD_FILTER) {
  int in = 32, ic = 32, ih = 128, iw = 128;
  int fn = 64, fc = 32, fh = 3, fw = 3;
  int on = 32, oc = 64, oh = 128, ow = 128;

  int ph = 1, pw = 1;
  int sh = 1, sw = 1;
  int dila_h = 1, dila_w = 1;

  int group = 1;

  std::vector<int> attrs = {in, ic, ih, iw, fn, fc, fh, fw, ph, pw, sh, sw, dila_h, dila_w, group, on, oc, oh, ow};
  /*
  absl::flat_hash_map<std::string, int> attrs = {
      {"input_n", in},   {"input_c", ic},   {"input_h", ih},        {"input_w", iw},        {"weights_n", fn},
      {"weights_c", fc}, {"weights_h", fh}, {"weights_w", fw},      {"pad_h", ph},          {"pad_w", pw},
      {"stride_h", sh},  {"stride_w", sw},  {"dilation_h", dila_h}, {"dilation_w", dila_w}, {"groups", group},
      {"output_n", on},  {"output_c", oc},  {"output_h", oh},       {"output_w", ow},
  };
  */

  CUDA_CALL(cudaSetDevice(0));
  auto buffer_x = common::BufferBuilder(Float(32), {in, ic, ih, iw}).set_random().Build();
  auto buffer_w = common::BufferBuilder(Float(32), {fn, fc, fh, fw}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {on, oc, oh, ow}).set_random().Build();

  void *dev_x = nullptr, *dev_w = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_w, buffer_w->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_w, buffer_w->memory, buffer_w->memory_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_y, buffer_y->memory, buffer_y->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _w;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _w.memory = static_cast<uint8_t*>(dev_w);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _w.memory_size = buffer_w->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x(&_x);
  cinn_pod_value_t w(&_w);
  cinn_pod_value_t y(&_y);

  Scope scope;
  auto target = common::DefaultNVGPUTarget();
  std::vector<std::string> in_args, out_args;
  TestInstruction instr(target, &scope, in_args, out_args, "conv2d");

  std::vector<cinn_pod_value_t> args = {x, y, w};
  std::vector<std::string> str_attrs = {"backward_filter"};
  instr.SetAttr(attrs);
  instr.SetStrAttr(str_attrs);
  instr.SetArgs(args);

  instr.RunX();

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_w));
  CUDA_CALL(cudaFree(dev_y));
}

#endif
}  // namespace framework
}  // namespace hlir
}  // namespace cinn

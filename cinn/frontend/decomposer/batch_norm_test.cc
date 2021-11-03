// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {
namespace {

struct Offset {
  int n;
  int c;
  int h;
  int w;

  Offset(int arg_n, int arg_c, int arg_h, int arg_w) : n(arg_n), c(arg_c), h(arg_h), w(arg_w) {}

  int operator()(int idx_n, int idx_c, int idx_h, int idx_w) const {
    return idx_n * c * h * w + idx_c * h * w + idx_h * w + idx_w;
  }
};

template <typename FuncType>
void Loop(FuncType func, const int n, const int c, const int h, const int w) {
  for (int in = 0; in < n; ++in) {
    for (int ic = 0; ic < c; ++ic) {
      for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < w; ++iw) {
          func(in, ic, ih, iw);
        }
      }
    }
  }
}

template <typename T>
void ComputeBatchNormTrainRef(const std::vector<T>& x,
                              const std::vector<T>& scale,
                              const std::vector<T>& bias,
                              const std::vector<T>& moving_mean,
                              const std::vector<T>& moving_variance,
                              const int n,
                              const int c,
                              const int h,
                              const int w,
                              std::vector<T>* y,
                              std::vector<T>* saved_mean,
                              std::vector<T>* saved_variance,
                              std::vector<T>* new_moving_mean,
                              std::vector<T>* new_moving_variance,
                              const float epsilon,
                              const float momentum) {
  Offset offset(n, c, h, w);

  // sum
  memset(saved_mean->data(), 0, sizeof(T) * c);
  auto func_sum_x = [=](int in, int ic, int ih, int iw) { saved_mean->at(ic) += x[offset(in, ic, ih, iw)]; };
  Loop(func_sum_x, n, c, h, w);

  // saved mean
  float element_count = static_cast<float>(n * h * w);
  for (int ic = 0; ic < c; ++ic) {
    // Checking result of saved_mean:
    // output[saved_mean], var_name=var_5, shape={32}
    // - Total 0 different results, offset=0, 0.00527001 vs 0.00527001, maximum_relative_diff=0(absolute_diff=0)
    saved_mean->at(ic) /= element_count;
  }

  // square_sum
  std::vector<float> x_square_mean(c, 0);
  auto func_sum_square_x = [&](int in, int ic, int ih, int iw) {
    x_square_mean.at(ic) += x[offset(in, ic, ih, iw)] * x[offset(in, ic, ih, iw)];
  };
  Loop(func_sum_square_x, n, c, h, w);

  for (int ic = 0; ic < c; ++ic) {
    x_square_mean[ic] /= element_count;
  }

  // saved variance, according to equation: E(x^2) - [E(x)]^2
  std::vector<float> std_variance(c);
  for (int ic = 0; ic < c; ++ic) {
    // Checking results of saved_variance and std_variance:
    // output[saved_variance], var_name=var_6, shape={32}
    // - Total 0 different results, offset=0, 0.336347 vs 0.336347, maximum_relative_diff=0(absolute_diff=0)
    // output[std_variance], var_name=std_variance, shape={32}
    // - Total 0 different results, offset=0, 0.579963 vs 0.579963, maximum_relative_diff=0(absolute_diff=0)
    saved_variance->at(ic) = x_square_mean[ic] - (saved_mean->at(ic) * saved_mean->at(ic));
    std_variance[ic]       = sqrt(saved_variance->at(ic) + epsilon);
  }

  // compute output
  std::vector<float> y_nobias(n * c * h * w);
  auto func_y_nobias = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    // Checking result of y_nobias:
    // output[y_nobias], var_name=y_nobias, shape={16, 32, 16, 16}
    // - Total 0 different results, offset=32104, -0.000488288 vs -0.000488288,
    // maximum_relative_diff=1.19208e-07(absolute_diff=5.82077e-11)
    y_nobias[idx] = (x[idx] - saved_mean->at(ic)) * scale[ic] / std_variance[ic];
  };
  Loop(func_y_nobias, n, c, h, w);

  auto func_y = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    // Checking result of y:
    // output[y], var_name=var_4, shape={16, 32, 16, 16}
    // - Total 80 different results, offset=126409, 1.81794e-06 vs 1.80304e-06,
    // maximum_relative_diff=0.00826446(absolute_diff=1.49012e-08) For the following case:
    //   idx=126409, y[idx]=1.80304e-06, y_nobias[idx]=0.2033332, bias[ic]=-0.2033314
    // The computing result of CPU and GPU may have some difference, like
    //   i=126409, 1.8179417e-06 vs 1.8030405e-06, relative_diff=0.0082644625, absolute_diff=1.4901161e-08
    // This case is considered reasonable.
    y->at(idx) = y_nobias[idx] + bias[ic];
  };
  Loop(func_y, n, c, h, w);

  // new moving runnning and variance
  float factor_0 = momentum;
  float factor_1 = static_cast<float>(1.0f - momentum);
  for (int ic = 0; ic < c; ++ic) {
    // Checking result of new_moving_mean and new_moving_variance:
    // output[new_moving_mean], var_name=var_7, shape={32}
    // - Total 0 different results, offset=9, 0.00123065 vs 0.00123065,
    // maximum_relative_diff=9.45967e-08(absolute_diff=1.16415e-10) output[new_moving_variance], var_name=var_8,
    // shape={32}
    // - Total 0 different results, offset=16, -0.00140787 vs -0.00140787,
    // maximum_relative_diff=5.29211e-06(absolute_diff=7.45058e-09)
    new_moving_mean->at(ic)     = moving_mean[ic] * factor_0 + saved_mean->at(ic) * factor_1;
    new_moving_variance->at(ic) = moving_variance[ic] * factor_0 + saved_variance->at(ic) * factor_1;
  }
}

TEST(Decomposer, BatchNormTrain) {
  // parameter
  int n = 16, c = 32, h = 16, w = 16;
  float epsilon           = 1e-5;
  float momentum          = 0.9f;
  std::string data_layout = "NCHW";
  NetBuilder net_builder("batch_norm_train");
  std::vector<std::string> output_names;
  {
    // create input
    auto x               = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale           = net_builder.CreateInput(Float(32), {c}, "scale");
    auto bias            = net_builder.CreateInput(Float(32), {c}, "bias");
    auto moving_mean     = net_builder.CreateInput(Float(32), {c}, "moving_mean");
    auto moving_variance = net_builder.CreateInput(Float(32), {c}, "moving_variance");

    // add batch norm train
    auto outputs =
        net_builder.batch_norm_train(x, scale, bias, moving_mean, moving_variance, epsilon, momentum, data_layout);
    for (auto output : outputs) {
      output_names.push_back(output->id);
    }
  }
  // build program
  auto program = net_builder.Build();

  auto target = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto run_program = gc.Build();

  // set input
  std::vector<float> x(n * c * h * w), scale(c), bias(c), moving_mean(c), moving_variance(c);
  InitRandomVector(&x, n * c * h * w);
  InitRandomVector(&scale, c);
  InitRandomVector(&bias, c);
  InitRandomVector(&moving_mean, c);
  InitRandomVector(&moving_variance, c);

  std::vector<float> y(n * c * h * w), new_moving_mean(c), new_moving_variance(c), saved_mean(c), saved_variance(c);
  ComputeBatchNormTrainRef<float>(x,
                                  scale,
                                  bias,
                                  moving_mean,
                                  moving_variance,
                                  n,
                                  c,
                                  h,
                                  w,
                                  &y,
                                  &saved_mean,
                                  &saved_variance,
                                  &new_moving_mean,
                                  &new_moving_variance,
                                  epsilon,
                                  momentum);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x}, {"scale", scale}, {"bias", bias}, {"moving_mean", moving_mean}, {"moving_variance", moving_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::unordered_map<std::string, std::pair<std::string, std::vector<float>>> outputs_ref = {
      {"new_moving_variance", {output_names[4], new_moving_variance}},
      {"new_moving_mean", {output_names[3], new_moving_mean}},
      {"saved_variance", {output_names[2], saved_variance}},
      {"saved_mean", {output_names[1], saved_mean}},
      {"y", {output_names[0], y}}};

  for (auto& iter : outputs_ref) {
    auto output = iter.second;
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);

    LOG(INFO) << "output[" << iter.first << "], var_name=" << output.first << ", shape=" << tensor->shape().data();
    if (iter.first == "y") {
      CheckOutput<float>(data, output.second, 1e-2, true);
    } else {
      CheckOutput<float>(data, output.second, 1e-5);
    }
  }
}

#if 0
template <typename T>
void cpu_batch_norm_grad(const std::vector<T>& x,
                         const std::vector<T>& dy,
                         const std::vector<T>& scale,
                         const std::vector<T>& save_mean,
                         const std::vector<T>& save_variance,
                         const int n,
                         const int c,
                         const int h,
                         const int w,
                         std::vector<T>* dx,
                         std::vector<T>* dscale,
                         std::vector<T>* dbias,
                         std::vector<T>* grad_std_norm,
                         std::vector<T>* grad_diff,
                         std::vector<T>* grad_std_variance_2d,
                         std::vector<T>* grad_variance_2d_without_mul,
                         std::vector<T>* grad_x0,
                         std::vector<T>* minus_grad_mean,
                         float epsilon = 1e-5) {
  Offset offset(n, c, h, w);

  std::vector<T> save_std_varance(c);
  for (int idx = 0; idx < c; ++idx) {
    save_std_varance[idx] = sqrt(save_variance[idx] + epsilon);
  }
  // grad bias
  memset(dbias->data(), 0, sizeof(float) * c);
  auto func_dbias = [=](int idx, int idy, int idz, int ida) {
    dbias->at(idy) += dy[offset(idx, idy, idz, ida)];
  };
  Loop(func_dbias, n, c, h, w);

  // grad scale
  memset(dscale->data(), 0, sizeof(float) * c);
  auto func_dscale = [=](int idx, int idy, int idz, int ida) {
    dscale->at(idy) += dy[offset(idx, idy, idz, ida)] *
                       ((x[offset(idx, idy, idz, ida)] - save_mean[idy]) / save_std_varance[idy]);
  };
  Loop(func_dscale, n, c, h, w);

  // grad_std
  auto func_grad_std_norm = [=](int idx, int idy, int idz, int ida) {
    grad_std_norm->at(offset(idx, idy, idz, ida)) = dy[offset(idx, idy, idz, ida)] * scale[idy];
  };
  Loop(func_grad_std_norm, n, c, h, w);

  auto func_grad_diff = [=](int idx, int idy, int idz, int ida) {
    grad_diff->at(offset(idx, idy, idz, ida)) =
        grad_std_norm->at(offset(idx, idy, idz, ida)) / save_std_varance[idy];
  };
  Loop(func_grad_diff, n, c, h, w);

  memset(grad_std_variance_2d->data(), 0, sizeof(float) * c);
  auto func_grad_std_variance_2d = [=](int idx, int idy, int idz, int ida) {
    grad_std_variance_2d->at(idy) += -1 * grad_std_norm->at(offset(idx, idy, idz, ida)) *
                                     (x[offset(idx, idy, idz, ida)] - save_mean[idy]) /
                                     (save_variance[idy] + epsilon);
  };
  Loop(func_grad_std_variance_2d, n, c, h, w);

  for (int idx = 0; idx < c; ++idx) {
    grad_variance_2d_without_mul->at(idx) = 0.5 * grad_std_variance_2d->at(idx) / save_std_varance[idx];
  }
  auto func_grad_x0 = [=](int idx, int idy, int idz, int ida) {
    grad_x0->at(offset(idx, idy, idz, ida)) =
        2 * x[offset(idx, idy, idz, ida)] * grad_variance_2d_without_mul->at(idy) / (n * h * w);
  };
  Loop(func_grad_x0, n, c, h, w);

  memset(minus_grad_mean->data(), 0, sizeof(float) * c);
  auto func_minus_grad_mean = [=](int idx, int idy, int idz, int ida) {
    minus_grad_mean->at(idy) += -1 * grad_diff->at(offset(idx, idy, idz, ida));
  };
  Loop(func_minus_grad_mean, n, c, h, w);

  for (int idx = 0; idx < c; ++idx) {
    minus_grad_mean->at(idx) += -1 * 2 * grad_variance_2d_without_mul->at(idx) * save_mean.at(idx);
    minus_grad_mean->at(idx) /= (n * h * w);
  }

  auto func_grad_x = [=](int idx, int idy, int idz, int ida) {
    dx->at(offset(idx, idy, idz, ida)) =
        grad_diff->at(offset(idx, idy, idz, ida)) +
        grad_x0->at(offset(idx, idy, idz, ida)) + minus_grad_mean->at(idy);
  };
  Loop(func_grad_x, n, c, h, w);
}

void GradX(const std::vector<float>& grad_std_norm,
           const std::vector<float>& x,
           const std::vector<float>& mean,
           const std::vector<float>& variance,
           int n,
           int c,
           int h,
           int w,
           float epsilon = 1e-5) {
  std::vector<float> std_variance(c);
  for (int idx = 0; idx < c; ++idx) {
    std_variance[idx] = sqrt(variance[idx] + epsilon);
  }

  std::vector<float> grad_diff(n * c * h * w);
  auto func_0 = [&](int idx, int idy, int idz, int ida) {
    grad_diff[idx * c * h * w + idy * h * w + idz * w + ida] =
        grad_std_norm[idx * c * h * w + idy * h * w + idz * w + ida] / std_variance[idy];
  };
  Loop(func_0, n, c, h, w);
  for (auto value : grad_diff) {
    std::cerr << value << " ";
  }
  std::cerr << std::endl;

  std::vector<float> grad_std_variance(c, 0);
  auto func_1 = [&](int idx, int idy, int idz, int ida) {
    grad_std_variance[idy] += -1 * grad_std_norm[idx * c * h * w + idy * h * w + idz * w + ida] *
                              (x[idx * c * h * w + idy * h * w + idz * w + ida] - mean[idy]) /
                              (variance[idy] + epsilon);
  };
  Loop(func_1, n, c, h, w);

  std::vector<float> grad_variance(c);
  for (int idx = 0; idx < c; ++idx) {
    grad_variance[idx] = grad_std_variance[idx] * 0.5 / std_variance[idx];
  }

  for (auto value : grad_variance) {
    std::cerr << value << " ";
  }
  std::cerr << std::endl;

  std::vector<float> grad_square_diff(n * c * h * w);
  auto func_11 = [&](int idx, int idy, int idz, int ida) {
    grad_square_diff[idx * c * h * w + idy * h * w + idz * w + ida] =
        grad_variance[idy] * 2 * (x[idx * c * h * w + idy * h * w + idz * w + ida] - mean[idy]) / float(n * h * w);
  };
  Loop(func_11, n, c, h, w);

  auto func_2 = [&](int idx, int idy, int idz, int ida) {
    grad_diff[idx * c * h * w + idy * h * w + idz * w + ida] +=
        grad_square_diff[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  Loop(func_2, n, c, h, w);

  std::vector<float> grad_mean(c, 0);
  auto func_3 = [&](int idx, int idy, int idz, int ida) {
    grad_mean[idy] += -1 * grad_diff[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  Loop(func_3, n, c, h, w);

  std::vector<float> grad_x(n * c * h * w);
  auto func_4 = [&](int idx, int idy, int idz, int ida) {
    grad_x[idx * c * h * w + idy * h * w + idz * w + ida] =
        grad_diff[idx * c * h * w + idy * h * w + idz * w + ida] + grad_mean[idy] / (float(n * h * w));
  };
  Loop(func_4, n, c, h, w);

  for (auto value : grad_x) {
    std::cerr << value << " ";
  }
  std::cerr << std::endl;
}

TEST(nn, BATCH_NORM_GRAD) {
  // parameter
  int n = 4, c = 16, h = 4, w = 4;
  int num = n * c * h * w;
  NetBuilder net_builder("net_builder_batch_norm_grad");
  std::vector<std::string> output_names;
  {
    // create input
    auto dy            = net_builder.CreateInput(Float(32), {n, c, h, w}, "dy");
    auto x             = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale         = net_builder.CreateInput(Float(32), {c}, "scale");
    auto save_mean     = net_builder.CreateInput(Float(32), {c}, "save_mean");
    auto save_variance = net_builder.CreateInput(Float(32), {c}, "save_variance");

    // add batch norm train
    auto outputs = net_builder.batch_norm_grad(dy, x, scale, save_mean, save_variance);
    for (auto output : outputs) {
      output_names.push_back(output->id);
    }
  }
  // build program
  auto program = net_builder.Build();

  auto target = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto run_program = gc.Build();

  // set input
  std::vector<float> dy(num), x(num), scale(c), save_mean(c, 0), save_variance(c, 0);
  InitRandomVector(&dy, num);
  InitRandomVector(&x, num);
  InitRandomVector(&scale, c);

  auto func_save_mean = [&](int idx, int idy, int idz, int ida) {
    save_mean[idy] += x[idx * c * h * w + idy * h * w + idz * w + ida];
    save_variance[idy] +=
        x[idx * c * h * w + idy * h * w + idz * w + ida] * x[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  Loop(func_save_mean, n, c, h, w);
  for (int idx = 0; idx < c; ++idx) {
    save_mean[idx] /= float(n * h * w);
    save_variance[idx] /= float(n * h * w);
    save_variance[idx] -= (save_mean[idx] * save_mean[idx]);
  }

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"dy", dy}, {"x", x}, {"scale", scale}, {"save_mean", save_mean}, {"save_variance", save_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::vector<float> dx(num), dscale(c), dbias(c);
  std::vector<float> grad_std_norm(num), grad_diff(num), grad_std_variance_1d(c), grad_variance_1d_without_mul(c),
      grad_x0(num), minus_grad_mean(c);

  cpu_batch_norm_grad(x,
                      dy,
                      scale,
                      save_mean,
                      save_variance,
                      n,
                      c,
                      h,
                      w,
                      &dx,
                      &dscale,
                      &dbias,
                      &grad_std_norm,
                      &grad_diff,
                      &grad_std_variance_1d,
                      &grad_variance_1d_without_mul,
                      &grad_x0,
                      &minus_grad_mean);
  // GradX(grad_std_norm, x, save_mean, save_variance, n, c, h , w);

  std::vector<std::pair<std::string, std::vector<float>>> outputs = {
      {output_names[2], dbias},
      {output_names[1], dscale},
      {output_names[0], dx},
  };

  for (auto& output : outputs) {
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);
    LOG(INFO) << output.first << " " << tensor->shape().numel();
    for (int idx = 0; idx < tensor->shape().numel(); ++idx) {
      float diff = abs((data[idx] - output.second[idx]) / data[idx]);
      if (diff > 1e-5) {
        LOG(INFO) << "i=" << idx << ", " << data[idx] << " vs " << output.second[idx] << ", diff=" << diff;
      }
//      ASSERT_LT(diff, 1e-3);
    }
  }
}
#endif

}  // namespace
}  // namespace frontend
}  // namespace cinn

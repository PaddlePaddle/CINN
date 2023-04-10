// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once

#include "tests/program_builder.h"

namespace cinn {
namespace tests {

class LayerNormBuilder : public ProgramBuilder {
 public:
  LayerNormBuilder() : ProgramBuilder("layer_norm_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo, const utils::AttributeMap& attrs = {}) {
    // int batch_size = inputs_varinfo[0][0];
    // int seq_len = inputs_varinfo[0][1];
    // int hidden_size = inputs_varinfo[0][2];
    int batch_size = 128;
    int seq_len = 512;
    int hidden_size = 768;
    // x
    auto A = builder_.CreateInput(Float(32), {batch_size, seq_len, hidden_size}, "A");
    // x * x
    auto B = builder_.Multiply(A, A);
    // sum x
    auto C = builder_.ReduceSum(A, {2});
    // sum x*x
    auto D = builder_.ReduceSum(B, {2});
    // constant w
    auto E = builder_.FillConstant<float>({batch_size, seq_len}, hidden_size, "E");
    // mean
    auto F  = builder_.Divide(C, E);
    auto FF = builder_.BroadcastTo(F, {batch_size, seq_len, hidden_size}, {0, 1});
    // output mean
    auto reshape_mean = builder_.Reshape(F, {65536});
    auto out_mean = builder_.Identity(reshape_mean);
    // mean x*x
    auto G = builder_.Divide(D, E);
    // mean * mean
    auto H = builder_.Multiply(F, F);
    // var^2
    auto I = builder_.Subtract(G, H);
    // output variance
    auto reshape_var = builder_.Reshape(I, {65536});
    auto out_var = builder_.Identity(reshape_var);
    // eps
    auto J = builder_.FillConstant<float>({batch_size, seq_len}, 1e-10f, "J");
    // eps + delta
    auto K = builder_.Add(I, J);
    // var
    auto L  = builder_.Sqrt(K);
    auto LL = builder_.BroadcastTo(L, {batch_size, seq_len, hidden_size}, {0, 1});
    // x - mean
    auto M = builder_.Subtract(A, FF);
    // /var
    auto N = builder_.Divide(M, LL);
    // weight
    auto weight = builder_.FillConstant<float>({hidden_size}, 1.0, "weight");
    auto ww = builder_.BroadcastTo(weight, {batch_size, seq_len, hidden_size}, {2});
    auto O = builder_.Multiply(N, ww);
    auto bias = builder_.FillConstant<float>({hidden_size}, 0.0, "bias");
    auto bb = builder_.BroadcastTo(bias, {batch_size, seq_len, hidden_size}, {2});
    auto P = builder_.Add(O, bb);
  }
};

/*
 * Add --* Multiply --* Add --* Relu
 */
class BiasBnReLUBuilder : public ProgramBuilder {
 public:
  BiasBnReLUBuilder() : ProgramBuilder("bias_bn_relu_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo, const utils::AttributeMap& attrs = {}) {
    CHECK(inputs_varinfo.size() == 4);
    auto conv_output = builder_.CreateInput(inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto bias        = builder_.CreateInput(inputs_varinfo[1].type, inputs_varinfo[1].shape, inputs_varinfo[1].id);
    auto bn_scale    = builder_.CreateInput(inputs_varinfo[2].type, inputs_varinfo[2].shape, inputs_varinfo[2].id);
    auto bn_offset   = builder_.CreateInput(inputs_varinfo[3].type, inputs_varinfo[3].shape, inputs_varinfo[3].id);

    auto bias_add = builder_.Add(conv_output, bias);
    auto bn_mul   = builder_.Multiply(bias_add, bn_scale);
    auto bn_add   = builder_.Add(bn_mul, bn_offset);
    builder_.Relu(bn_add);
    return builder_.Build();
  }
};

/*
 * Exp --* Add
 *    \
 *     --* Multiply
 */
class ExpTwoConsumersOpBuilder : public ProgramBuilder {
 public:
  ExpTwoConsumersOpBuilder() : ProgramBuilder("exp_two_consumers_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo, const utils::AttributeMap& attrs = {}) {
    CHECK(inputs_varinfo.size() == 1);
    auto x     = builder_.CreateInput(inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto exp_x = builder_.Exp(x);
    auto add_x = builder_.Add(exp_x, x);
    auto mul_1 = builder_.Multiply(exp_x, add_x);
    return builder_.Build();
  }
};

/*
 * Gather --* Add --* Subtract
 *                    *
 *                   /
 *            Gather
 */
class GatherAddSubBuilder : public ProgramBuilder {
 public:
  GatherAddSubBuilder() : ProgramBuilder("gather_add_sub_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo, const utils::AttributeMap& attrs = {}) {
    CHECK(inputs_varinfo.size() == 2);
    auto x             = builder_.CreateInput(inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto y             = builder_.CreateInput(inputs_varinfo[1].type, inputs_varinfo[1].shape, inputs_varinfo[1].id);
    auto input_x_shape = inputs_varinfo[0].shape;
    auto where_x_0     = builder_.Gather(x, builder_.FillConstant({input_x_shape[0]}, 0, "constant_idx_first"));
    auto where_x_last =
        builder_.Gather(x, builder_.FillConstant({input_x_shape[0]}, input_x_shape[0] - 1, "constant_idx_last"));
    auto add_1 = builder_.Add(where_x_0, y);
    builder_.Subtract(where_x_last, add_1);
    return builder_.Build();
  }
};

/*
 * FillConstant --* Add
 */
class FillConstantAddBuilder : public ProgramBuilder {
 public:
  FillConstantAddBuilder() : ProgramBuilder("fill_constant_add_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo, const utils::AttributeMap& attrs = {}) {
    CHECK(inputs_varinfo.size() == 1);
    auto x             = builder_.CreateInput(inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto fill_constant = builder_.FillConstant(inputs_varinfo[0].shape, 1.0f, "fill_constant");
    builder_.Add(x, fill_constant);
    return builder_.Build();
  }
};

}  // namespace tests
}  // namespace cinn

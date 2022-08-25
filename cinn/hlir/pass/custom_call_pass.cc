// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/common/type.h"
#include "cinn/hlir/pass/fusion_helper_base.h"
namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using ShapeDict         = absl::flat_hash_map<std::string, shape_t>;
using ConditionFunction = std::function<bool(const Node*, const Node*)>;

class GraphAlterHelper {
 public:
  GraphAlterHelper(Graph* graph) : graph_(graph) {}
  void MatmulToCustomCall() {
    auto nodes = graph_->CollectNodes([](const common::GraphNode* graph_node) -> bool {
      if (graph_node->safe_as<Node>()) {
        auto node = graph_node->safe_as<Node>();
        if (node->op()->name == "matmul" || node->op()->name == "mul" || node->op()->name == "cublas_gemm" ||
            node->op()->name == "cublas_matmul") {
          reutrn true;
        }
      }

      return false;
    });

    for (auto gnode : nodes) {
      auto node = gnode->safe_as<Node>();
      CHECK(node);
      auto dst                             = GetCustomCallNode(node);
      dst->attrs.attr_store["custom_call"] = "cinn_call_cublas";
      Alter(node, dst);
    }
  }

  void ConvToCustomCall() {
    auto nodes = graph_->CollectNodes([](const common::GraphNode* graph_node) -> bool {
      if (graph_node->safe_as<Node>()) {
        auto node = graph_node->safe_as<Node>();
        if (node->op()->name == "conv2d") {
          reutrn true;
        }
      }

      return false;
    });

    for (auto gnode : nodes) {
      auto node = gnode->safe_as<Node>();
      CHECK(node);
      auto dst = GetCustomCallNode(node);
      CHECK(dst->attrs.attr_store.count("conv_type"));
      auto type = dst->attrs.attr_store.count("conv_type") ? std::get<std::string>(dst->attrs.attr_store["conv_type"])
                                                           : "forward";
      switch (type) {
        case "forward":
          dst->attrs.attr_store["custom_call"] = "cinn_call_cudnn_conv2d_forward";
          break;
        case "backward_data":
          dst->attrs.attr_store["custom_call"] = "cinn_call_cudnn_conv2d_backward_data";
          break;
        case "backward_filter":
          dst->attrs.attr_store["custom_call"] = "cinn_gpu_cudnn_conv2d_backward_filter";
          break;
        default:
          LOG(FATAL) << "conv type is unkown!";
      }
      Alter(node, dst);
    }
  }

 private:
  void Alter(Node* src, Node* dst) {
    // input to src
    for (auto& edge : src->inlinks_in_order()) {
      auto input_data = edge->source()->safe_as<NodeData>();
      CHECK(input_data);

      input_data.UnLinkSingleTo(src);
      input_data.LinkTo(dst);
    }

    // src to output
    for (auto& edge : src->outlinks_in_order()) {
      auto output_data = edge->sink()->safe_as<NodeData>();
      CHECK(output_data);

      src.UnLinkSingleTo(output_data);
      dst.LinkTo(output_data);
    }
  }
  Node* GetCustomCallNode(Node* src) {
    auto dst   = new Node(framework::Operator::Get("custom_call"), src->attrs.node_name, src->id());
    dst->attrs = src->attrs;
    return dst;
  }

  Graph* graph_;
};

void MatmulToCustomCallPassInternal(Graph* graph) {
  VLOG(3) << "OpFusionPass...!";
  GraphAlterHelper(graph).MatmulToCustomCall();
  VLOG(3) << "OpFusionPass Finish...!";
}

void ConvToCustomCallPassInternal(Graph* graph) {
  VLOG(3) << "OpFusionPass...!";
  GraphAlterHelper(graph).ConvToCustomCall();
  VLOG(3) << "OpFusionPass Finish...!";
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(CustomCallPass) {
  CINN_REGISTER_PASS(MatmulToCustomCallPass)
      .describe("This pass which convert matmul op to custom call pass.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::MatmulToCustomCallPassInternal);

  CINN_REGISTER_PASS(ConvToCustomCallPass)
      .describe("This pass which convert conv op to custom call pass.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::ConvToCustomCallPassInternal);

  return true;
}

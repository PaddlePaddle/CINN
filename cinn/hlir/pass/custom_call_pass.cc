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
          return true;
        }
      }

      return false;
    });

    for (auto gnode : nodes) {
      auto src = gnode->safe_as<Node>();
      CHECK(src);
      auto dst                             = GetCustomCallNode(src);
      dst->attrs.attr_store["custom_call"] = std::string("cinn_call_cublas");
      Alter(src, dst);
    }
  }

  void ConvToCustomCall() {
    auto nodes = graph_->CollectNodes([](const common::GraphNode* graph_node) -> bool {
      if (graph_node->safe_as<Node>()) {
        auto node = graph_node->safe_as<Node>();
        if (node->op()->name == "conv2d") {
          return true;
        }
      }

      return false;
    });

    for (auto gnode : nodes) {
      auto src = gnode->safe_as<Node>();
      CHECK(src);
      auto dst = GetCustomCallNode(src);
      CHECK(dst->attrs.attr_store.count("conv_type"));
      std::string type = dst->attrs.attr_store.count("conv_type") ? absl::get<std::string>(dst->attrs.attr_store["conv_type"])
                                                           : "forward";
      if(type == "forward") {
        dst->attrs.attr_store["custom_call"] = std::string("cinn_call_cudnn_conv2d_forward");
      } else if(type == "backward_data") {
        dst->attrs.attr_store["custom_call"] = std::string("cinn_call_cudnn_conv2d_backward_data");
      } else if(type == "backward_filter") {
        dst->attrs.attr_store["custom_call"] = std::string("cinn_gpu_cudnn_conv2d_backward_filter");
      } else {
        LOG(FATAL) << "conv type is unkown!";
      }
      Alter(src, dst);
    }
  }

 private:
  void Alter(Node* src, Node* dst) {
    // input to src
    for (auto& edge : src->inlinks_in_order()) {
      auto input_data = edge->source()->safe_as<NodeData>();
      CHECK(input_data);

      input_data->UnLinkSingleTo(src);
      input_data->LinkTo(dst);
    }

    // src to output
    for (auto& edge : src->outlinks_in_order()) {
      auto output_data = edge->sink()->safe_as<NodeData>();
      CHECK(output_data);

      src->UnLinkSingleTo(output_data);
      dst->LinkTo(output_data);
    }

    graph_->DropNode(src);
  }
  Node* GetCustomCallNode(Node* src) {
    auto dst   = new Node(framework::Operator::Get("custom_call"), src->attrs.node_name, src->id());
    graph_->RegisterNode(dst->id(), dst);
    dst->attrs.attr_store = src->attrs.attr_store;
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

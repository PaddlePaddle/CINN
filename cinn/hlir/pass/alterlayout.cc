#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/layout.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpValueType;

using InferShapeFunc = std::function<std::vector<framework::shape_t>(
    const std::vector<framework::shape_t>&, const framework::NodeAttr&, const Target&)>;
using InferTypeFunc =
    std::function<std::vector<Type>(const std::vector<Type>&, const framework::NodeAttr&, const Target&)>;
using InferLayoutFunc = std::function<std::vector<std::vector<std::string>>(const std::vector<framework::shape_t>&,
                                                                            const std::vector<std::string>&,
                                                                            const framework::NodeAttr&,
                                                                            const Target&)>;

std::tuple<Node*, NodeData*> InsertLayoutTransformNode(Graph* graph,
                                                       NodeData* input_data,
                                                       Node* dst_node,
                                                       const std::string& src_layout,
                                                       const std::string& dst_layout,
                                                       const std::string& name) {
  CHECK(graph);
  CHECK(input_data);
  std::string op_type                           = "layout_transform";
  auto trans_node                               = new Node(Operator::Get(op_type), op_type, name);
  trans_node->attrs.attr_store["src_layout"]    = src_layout;
  trans_node->attrs.attr_store["dst_layout"]    = dst_layout;
  auto output_data                              = InsertGraphOpNode(graph, trans_node, input_data, dst_node);
  trans_node->attrs.attr_store["input_layouts"] = {src_layout};
  trans_node->attrs.attr_store["out_layouts"]   = {dst_layout};
  return std::make_tuple(trans_node, output_data);
}

std::vector<framework::shape_t> updateInferInfos(Node* node,
                                                 const std::vector<framework::shape_t>& input_shapes,
                                                 const std::vector<Type>& input_types,
                                                 const std::vector<std::string>& input_layouts,
                                                 const common::Target& target,
                                                 const OpValueType<InferShapeFunc>& op_infershape,
                                                 const OpValueType<InferTypeFunc>& op_infertype,
                                                 const OpValueType<InferLayoutFunc>& op_inferlayout,
                                                 std::unordered_map<std::string, framework::shape_t>* shape_dict,
                                                 std::unordered_map<std::string, Type>* type_dict,
                                                 std::unordered_map<std::string, std::string>* layout_dict) {
  CHECK(shape_dict);
  CHECK(type_dict);
  CHECK(layout_dict);
  CHECK(op_infershape[node->op()]) << "find no InferShape function for op " << node->op()->name;
  CHECK(op_infertype[node->op()]) << "find no InferDtype function for op " << node->op()->name;
  CHECK(op_inferlayout[node->op()]) << "find no InferLayout function for op " << node->op()->name;
  auto infershapes  = op_infershape[node->op()](input_shapes, node->attrs, target);
  auto infertypes   = op_infertype[node->op()](input_types, node->attrs, target);
  auto inferlayouts = op_inferlayout[node->op()](input_shapes, input_layouts, node->attrs, target);

  CHECK(!infershapes.empty()) << node->op()->name << " finds no infershape";
  CHECK(!infertypes.empty()) << node->op()->name << " finds no infertype";
  CHECK(!inferlayouts.empty()) << node->op()->name << " finds no inferlayout";
  auto outlinks = node->outlinks_in_order(true);
  // check opt
  CHECK(infershapes.size() == infertypes.size());
  CHECK_EQ(inferlayouts.size(), 2U);
  CHECK(infertypes.size() == inferlayouts[0].size());
  CHECK_EQ(outlinks.size(), infershapes.size());

  for (int i = 0; i < outlinks.size(); i++) {
    auto* sink                 = outlinks[i]->sink();
    (*shape_dict)[sink->id()]  = infershapes[i];
    (*type_dict)[sink->id()]   = infertypes[i];
    (*layout_dict)[sink->id()] = inferlayouts[0][i];
    LOG(INFO) << "Infershape: " << sink->id() << " " << utils::Join(infershapes[i], ", ");
  }
  node->attrs.attr_store["out_layouts"]   = inferlayouts[0];
  node->attrs.attr_store["input_layouts"] = inferlayouts[1];
  return infershapes;
}

void AlterLayoutPass(Graph* graph) {
  // alterlayout only in X86 for it's specific layout requirements
  if (graph->target_.arch == Target::Arch::X86) {
    auto store_nodes     = std::get<0>(graph->topological_order());
    auto& shape_dict     = graph->GetMutableAttrs<std::unordered_map<std::string, framework::shape_t>>("infershape");
    auto& type_dict      = graph->GetMutableAttrs<std::unordered_map<std::string, Type>>("inferdtype");
    auto& op_infershape  = Operator::GetAttrs<InferShapeFunc>("infershape");
    auto& op_inferdtype  = Operator::GetAttrs<InferTypeFunc>("inferdtype");
    auto& op_inferlayout = Operator::GetAttrs<InferLayoutFunc>("inferlayout");
    std::unordered_map<std::string, std::string> layout_dict;

    bool has_altered = false;
    for (int i = 0; i < store_nodes.size(); i++) {
      auto node = store_nodes[i]->safe_as<Node>();
      if (node) {
        if (node->op()->name == "conv2d") {
          CHECK(node->attrs.attr_store.count("data_format")) << node->op()->name << " op has no data_format attr";
          std::string data_format = std::get<std::string>(node->attrs.attr_store.at("data_format"));
          if (data_format != "NCHW") {
            // not NCHW such as NHWC or has already been altered layout
            continue;
          }
          has_altered             = true;
          std::string new_op_type = node->op()->name + "_NCHWc";
          // alter conv2d op to conv2d_NCHWc
          Node* new_node             = new Node(Operator::Get(new_op_type), new_op_type, common::UniqName(new_op_type));
          new_node->attrs.attr_store = node->attrs.attr_store;
          std::string new_data_format               = "NCHWc";
          new_node->attrs.attr_store["data_format"] = new_data_format;

          auto conv_inlinks = node->inlinks_in_order(true);
          std::vector<common::GraphNode*> input_nodes;
          for (auto& link : conv_inlinks) {
            auto* source = link->source();
            input_nodes.push_back(source);
          }
          // get new layout: ic_bn, oc_bn
          CHECK_EQ(input_nodes.size(), 2U) << "conv2d should have 2 input nodes";
          auto* input_node  = input_nodes[0];
          auto* weight_node = input_nodes[1];
          CHECK(shape_dict.count(input_node->id())) << input_node->id() << " has no infershape";
          CHECK(shape_dict.count(weight_node->id())) << weight_node->id() << " has no infershape";
          CHECK(type_dict.count(input_node->id())) << input_node->id() << " has no infertype";
          CHECK(type_dict.count(weight_node->id())) << weight_node->id() << " has no infertype";
          auto input_shape  = shape_dict.at(input_node->id());
          auto weight_shape = shape_dict.at(weight_node->id());
          auto input_type   = type_dict.at(input_node->id());
          auto weight_type  = type_dict.at(weight_node->id());
          Node* weight_trans_node;
          Node* input_trans_node;
          std::vector<framework::shape_t> conv2d_NCHWc_inputshapes;
          std::vector<Type> conv2d_NCHWc_inputtypes;
          std::vector<std::string> conv2d_NCHWc_inputlayouts;
          CHECK(weight_shape.size() == 4) << "old conv2d's weight shape should be 4";
          std::unordered_map<std::string, int> conv2d_factors;
          int oc = weight_shape[0];
          int fc = weight_shape[1];
          int ic = input_shape[1];
          if (input_shape.size() == 4) {
            pe::GetConv2dFactors(&conv2d_factors, oc, ic, -1, input_type, graph->target_);
            int ic_bn                    = conv2d_factors["ic_bn"];
            std::string src_input_layout = "NCHW";
            std::string dst_input_layout = "NCHW" + std::to_string(ic_bn) + "c";
            LOG(INFO) << "dst_input_layout: " << dst_input_layout;

            // insert input layout_transform
            auto input_data = input_node->safe_as<NodeData>();
            CHECK(input_data);
            NodeData* output_data;
            std::tie(input_trans_node, output_data) =
                InsertLayoutTransformNode(graph,
                                          input_data,
                                          node,
                                          src_input_layout,
                                          dst_input_layout,
                                          common::UniqName(node->op()->name + "_input_layout_tranform"));
            updateInferInfos(input_trans_node,
                             {input_shape},
                             {input_type},
                             {src_input_layout},
                             graph->target_,
                             op_infershape,
                             op_inferdtype,
                             op_inferlayout,
                             &shape_dict,
                             &type_dict,
                             &layout_dict);
            CHECK(shape_dict.count(output_data->id())) << output_data->id() << " finds no infershape in shape_dict.";
            CHECK(type_dict.count(output_data->id())) << output_data->id() << " finds no infertype in shape_dict.";
            auto trans_out_shapes = shape_dict[output_data->id()];
            auto trans_out_dtypes = type_dict[output_data->id()];
            conv2d_NCHWc_inputshapes.push_back(trans_out_shapes);
            conv2d_NCHWc_inputtypes.push_back(trans_out_dtypes);
            conv2d_NCHWc_inputlayouts.push_back(dst_input_layout);
          } else {
            CHECK_EQ(input_shape.size(), 5U) << "conv2d_NCHWc op's input shape dim should be 5";
            conv2d_NCHWc_inputshapes.push_back(input_shape);
            conv2d_NCHWc_inputtypes.push_back(input_type);
            CHECK(layout_dict.count(input_node->id())) << input_node->id() << " should have out_layout attr";
            conv2d_NCHWc_inputlayouts.push_back(layout_dict[input_node->id()]);
          }
          if (weight_shape.size() == 4) {
            // opt: alterlayout func attr?
            pe::GetConv2dFactors(&conv2d_factors, oc, fc, -1, input_type, graph->target_);
            int fc_bn                     = conv2d_factors["ic_bn"];
            int oc_bn                     = conv2d_factors["oc_bn"];
            std::string src_kernel_layout = "OIHW";
            std::string dst_kernel_layout = "OIHW" + std::to_string(fc_bn) + "i" + std::to_string(oc_bn) + "o";
            LOG(INFO) << "dst_kernel_layout: " << dst_kernel_layout;
            // insert weight layout_transform
            auto weight_data = weight_node->safe_as<NodeData>();
            CHECK(weight_data);
            NodeData* output_data;
            std::tie(weight_trans_node, output_data) =
                InsertLayoutTransformNode(graph,
                                          weight_data,
                                          node,
                                          src_kernel_layout,
                                          dst_kernel_layout,
                                          common::UniqName(node->op()->name + "_weight_layout_tranform"));
            updateInferInfos(weight_trans_node,
                             {weight_shape},
                             {weight_type},
                             {src_kernel_layout},
                             graph->target_,
                             op_infershape,
                             op_inferdtype,
                             op_inferlayout,
                             &shape_dict,
                             &type_dict,
                             &layout_dict);

            CHECK(shape_dict.count(output_data->id())) << output_data->id() << " finds no infershape in shape_dict.";
            CHECK(type_dict.count(output_data->id())) << output_data->id() << " finds no infertype in shape_dict.";
            auto trans_out_shapes = shape_dict[output_data->id()];
            auto trans_out_dtypes = type_dict[output_data->id()];
            conv2d_NCHWc_inputshapes.push_back(trans_out_shapes);
            conv2d_NCHWc_inputtypes.push_back(trans_out_dtypes);
            conv2d_NCHWc_inputlayouts.push_back(dst_kernel_layout);
          } else {
            CHECK_EQ(weight_shape.size(), 6U) << weight_node->id() << " shape dim should be 6";
            conv2d_NCHWc_inputshapes.push_back(weight_shape);
            conv2d_NCHWc_inputtypes.push_back(weight_type);
            CHECK(layout_dict.count(weight_node->id())) << weight_node->id() << " should have out_layout attr";
            conv2d_NCHWc_inputlayouts.push_back(layout_dict[weight_node->id()]);
          }
          // replace conv2d to conv2d_NCHWc
          auto infershapes = op_infershape[new_node->op()](conv2d_NCHWc_inputshapes, new_node->attrs, graph->target_);
          LOG(INFO) << "out_size " << infershapes.size();
          framework::ReplaceGraphOpNode(graph, node, new_node, infershapes.size());
          // update conv2d_NCHWc's infershape, infertype, inferlayout and set attrs
          updateInferInfos(new_node,
                           conv2d_NCHWc_inputshapes,
                           conv2d_NCHWc_inputtypes,
                           conv2d_NCHWc_inputlayouts,
                           graph->target_,
                           op_infershape,
                           op_inferdtype,
                           op_inferlayout,
                           &shape_dict,
                           &type_dict,
                           &layout_dict);

        } else if (has_altered) {
          // not alterlayout like conv2d, just inferlayout
          // opt: to a func, updateInferInfos
          // update infershape, infertype, inferlayout?
          std::vector<framework::shape_t> input_shapes;
          std::vector<Type> input_types;
          std::vector<std::string> input_layouts;
          for (auto& link : node->inlinks_in_order(true)) {
            auto* source = link->source();
            CHECK(shape_dict.count(source->id())) << source->id() << " finds no infershape";
            CHECK(type_dict.count(source->id())) << source->id() << " finds no infertype";
            // CHECK(layout_dict.count(source->id()))<<source->id()<<" finds no inferlayout";
            input_shapes.push_back(shape_dict[source->id()]);
            input_types.push_back(type_dict[source->id()]);
            if (layout_dict.count(source->id())) {
              input_layouts.push_back(layout_dict[source->id()]);
            } else {
              input_layouts.push_back("");
            }
          }
          updateInferInfos(node,
                           input_shapes,
                           input_types,
                           input_layouts,
                           graph->target_,
                           op_infershape,
                           op_inferdtype,
                           op_inferlayout,
                           &shape_dict,
                           &type_dict,
                           &layout_dict);
          // if input inferred layouts is different from original's, expand dims or do transformation.
          CHECK(node->attrs.attr_store.count("input_layouts")) << node->id() << " find no input_layouts attr";
          auto new_input_layouts = std::get<std::vector<std::string>>(node->attrs.attr_store["input_layouts"]);
          auto inlinks           = node->inlinks_in_order();
          CHECK_EQ(input_layouts.size(), inlinks.size());
          CHECK_EQ(input_layouts.size(), new_input_layouts.size());
          CHECK_EQ(input_layouts.size(), input_shapes.size());
          for (int i = 0; i < inlinks.size(); i++) {
            if (input_layouts[i] != new_input_layouts[i]) {
              // expand dims or do transformation
              int input_shape_size = input_shapes[i].size();
              if (input_shape_size == 1 && new_input_layouts[i].size() > 4) {
                // C -> NCHWxc: 1. C -> NCHW 2. layout transform from NCHW to NCHWxc
                int axis = -1;
                CHECK(node->attrs.attr_store.count("axis")) << node->id() << " find no axis attr";
                axis = std::get<int>(node->attrs.attr_store["axis"]);
                // must check?
                // if (new_dims > 4) {
                CHECK(new_input_layouts[i].substr(0, 4) == "NCHW") << "only support NCHWxc";
                if (axis == -1) {
                  axis += 4;
                }
                std::vector<int> new_shapes;
                for (int j = 0; j < 4; j++) {
                  if (axis == j) {
                    new_shapes.push_back(input_shapes[i][0]);
                  } else {
                    new_shapes.push_back(1);
                  }
                }
                auto source               = inlinks[i]->source();
                std::string src_layout    = "NCHW";
                shape_dict[source->id()]  = new_shapes;
                layout_dict[source->id()] = src_layout;
                // check
                node->attrs.attr_store["axis"] = -1;
                // insert layout tranfrom
                auto input_data = source->safe_as<NodeData>();
                CHECK(input_data);
                NodeData* output_data;
                Node* trans_node;
                LOG(INFO) << source->id() << " do layout_tranform from C to NCHWxc";
                std::tie(trans_node, output_data) =
                    InsertLayoutTransformNode(graph,
                                              input_data,
                                              node,
                                              src_layout,
                                              new_input_layouts[i],
                                              common::UniqName(source->id() + "_layout_tranform"));
                LOG(INFO) << graph->Visualize();
                updateInferInfos(trans_node,
                                 {new_shapes},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              } else if (input_shape_size == 4 && new_input_layouts[i].size() > 4) {
                // NCHW -> NCHWxc
                // insert layout tranfrom
                // opt: duplicate code
                auto source               = inlinks[i]->source();
                auto src_layout           = "NCHW";
                layout_dict[source->id()] = src_layout;
                auto input_data           = source->safe_as<NodeData>();
                CHECK(input_data);
                NodeData* output_data;
                Node* trans_node;
                LOG(INFO) << source->id() << " do layout_tranform from NCHW to NCHWxc";
                std::tie(trans_node, output_data) =
                    InsertLayoutTransformNode(graph,
                                              input_data,
                                              node,
                                              src_layout,
                                              new_input_layouts[i],
                                              common::UniqName(source->id() + "_layout_tranform"));
                updateInferInfos(trans_node,
                                 {input_shapes[i]},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              } else if (input_shape_size == 5 && new_input_layouts[i].size() == 4) {
                // NCHWxc -> NCHW
                // insert layout tranfrom
                auto source               = inlinks[i]->source();
                auto src_layout           = input_layouts[i];
                layout_dict[source->id()] = src_layout;
                auto input_data           = source->safe_as<NodeData>();
                CHECK(input_data);
                NodeData* output_data;
                Node* trans_node;
                LOG(INFO) << source->id() << " do layout_tranform from NCHWxc to NCHW";
                std::tie(trans_node, output_data) =
                    InsertLayoutTransformNode(graph,
                                              input_data,
                                              node,
                                              src_layout,
                                              new_input_layouts[i],
                                              common::UniqName(source->id() + "_layout_tranform"));
                updateInferInfos(trans_node,
                                 {input_shapes[i]},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              }
            }
          }
        }
      }
    }
    // opt: last node? in forloop
    LOG(INFO) << graph->Visualize();
    store_nodes = std::get<0>(graph->topological_order());
    for (int i = store_nodes.size() - 1; i >= 0; i--) {
      auto* node = store_nodes[i]->safe_as<Node>();

      if (node) {
        CHECK(node->attrs.attr_store.count("out_layouts")) << node->id() << " finds no out_layouts attr";
        auto out_layouts = std::get<std::vector<std::string>>(node->attrs.attr_store.at("out_layouts"));
        CHECK(!out_layouts.empty());
        if (out_layouts[0].size() > 4) {
          // recover the layout finally, NCHWxc->NCHW, only first output
          auto outlinks = node->outlinks_in_order(true);
          CHECK(!outlinks.empty());
          auto* out_node         = outlinks[0]->sink();
          std::string dst_layout = "NCHW";
          CHECK(layout_dict.count(out_node->id())) << out_node->id() << " finds no out_layout";
          std::string src_layout = layout_dict[out_node->id()];
          // insert layout_transform
          NodeData* output_data;
          Node* trans_node;
          std::tie(trans_node, output_data) =
              InsertLayoutTransformNode(graph,
                                        out_node->safe_as<NodeData>(),
                                        nullptr,
                                        src_layout,
                                        dst_layout,
                                        common::UniqName(node->op()->name + "_final_layout_tranform"));

          // update layout_transform's infershape, infertype, inferlayout
          CHECK(shape_dict.count(out_node->id())) << out_node->id() << " finds no infershape";
          CHECK(type_dict.count(out_node->id())) << out_node->id() << " finds no infertype";
          auto shape = shape_dict[out_node->id()];
          auto type  = type_dict[out_node->id()];
          updateInferInfos(trans_node,
                           {shape},
                           {type},
                           {src_layout},
                           graph->target_,
                           op_infershape,
                           op_inferdtype,
                           op_inferlayout,
                           &shape_dict,
                           &type_dict,
                           &layout_dict);
        }
        break;
      }
    }
    graph->ClearUnlinkedNodes(&shape_dict, &type_dict, &layout_dict);
    graph->attrs["infershape"]  = std::make_shared<std::any>(shape_dict);
    graph->attrs["inferdtype"]  = std::make_shared<std::any>(type_dict);
    graph->attrs["inferlayout"] = std::make_shared<std::any>(layout_dict);
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(AlterLayout) {
  CINN_REGISTER_PASS(AlterLayout)
      .describe(
          "This pass alters ops' data layouts in the graph(e.g. NCHW -> NCHWxc, OIHW -> OIHWxoxi) and saves to "
          "g.attrs[\"inferlayout\"]")
      .set_change_structure(true)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::AlterLayoutPass);
  return true;
}

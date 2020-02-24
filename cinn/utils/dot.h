#pragma once
#include <glog/logging.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cinn {
namespace utils {

static size_t dot_node_counter{0};

struct Node;
struct Edge;
struct Attr;
/*
 * A Dot template that helps to build a DOT graph definition.
 */
class Dot {
 public:
  Dot() = default;

  explicit Dot(const std::vector<Attr>& attrs) : attrs_(attrs) {}
  /**
   * Add a node ot DOT graph.
   * @param id Unique ID for this node.
   * @param attrs DOT attributes.
   * @param label Name of the node.
   */

  void AddNode(const std::string& id, const std::vector<Attr>& attrs, std::string label = "");

  /**
   * Add an edge to the DOT graph.
   * @param source The id of the source of the edge.
   * @param target The id of the sink of the edge.
   * @param attrs The attributes of the edge.
   */
  void AddEdge(const std::string& source, const std::string& target, const std::vector<Attr>& attrs);

  std::string operator()() const { return Build(); }

 private:
  // Compile to DOT language codes.
  std::string Build() const;

  std::map<std::string, Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Attr> attrs_;
};

}  // namespace utils
}  // namespace cinn

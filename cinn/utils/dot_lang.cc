#include "cinn/utils/dot_lang.h"

namespace cinn {
namespace utils {

std::string Attr::repr() const {
  std::stringstream ss;
  ss << key << "=" << '"' << value << '"';
  return ss.str();
}

Node::Node(const std::string& name, const std::vector<Attr>& attrs) : name(name), attrs(attrs) {
  std::stringstream ss;
  ss << "node_" << dot_node_counter++;
  id_ = ss.str();
}
std::string Node::repr() const {
  std::stringstream ss;
  CHECK(!name.empty());
  ss << id_;
  if (attrs.empty()) {
    ss << "[label=" << '"' << name << '"' << "]";
    return ss.str();
  }
  for (size_t i = 0; i < attrs.size(); i++) {
    if (i == 0) {
      ss << "[label=" << '"' << name << '"' << " ";
    }
    ss << attrs[i].repr();
    ss << ((i < attrs.size() - 1) ? " " : "]");
  }
  return ss.str();
}
std::string Edge::repr() const {
  std::stringstream ss;
  CHECK(!source.empty());
  CHECK(!target.empty());
  ss << source << "->" << target;
  for (size_t i = 0; i < attrs.size(); i++) {
    if (i == 0) {
      ss << "[";
    }
    ss << attrs[i].repr();
    ss << ((i < attrs.size() - 1) ? " " : "]");
  }
  return ss.str();
}
void DotLang::AddNode(const std::string& id, const std::vector<Attr>& attrs, std::string label) {
  CHECK(!nodes_.count(id)) << "duplicate Node '" << id << "'";
  if (label.empty()) label = id;
  nodes_.emplace(id, Node{label, attrs});
}
void DotLang::AddEdge(const std::string& source, const std::string& target, const std::vector<Attr>& attrs) {
  CHECK(!source.empty());
  CHECK(!target.empty());
  auto sid = nodes_.at(source).id();
  auto tid = nodes_.at(target).id();
  edges_.emplace_back(sid, tid, attrs);
}
std::string DotLang::Build() const {
  std::stringstream ss;
  const std::string indent = "   ";
  ss << "digraph G {" << '\n';

  // Add graph attrs
  for (const auto& attr : attrs_) {
    ss << indent << attr.repr() << '\n';
  }
  // add nodes
  for (auto& item : nodes_) {
    ss << indent << item.second.repr() << '\n';
  }
  // add edges
  for (auto& edge : edges_) {
    ss << indent << edge.repr() << '\n';
  }
  ss << "} // end G";
  return ss.str();
}

}  // namespace utils
}  // namespace cinn

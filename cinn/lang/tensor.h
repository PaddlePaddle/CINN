#pragma once

#include <isl/cpp.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/function_base.h"
#include "cinn/lang/buffer.h"

namespace cinn {

namespace poly {
struct Stage;
}  // namespace poly

namespace lang {
template <typename T>
struct Placeholder;
}  // namespace lang

namespace ir {
namespace detail {
constexpr bool LE(int a, int b) { return a <= b; }
constexpr bool GE(int a, int b) { return a >= b; }

}  // namespace detail

class _Tensor_;

/**
 * @brief Tensor representing a possible input, or intermediate computation result.
 *
 * Tensor is the a general type in CINN, it holds a computation(analygous to a node in SSA graph) or placeholder. A
 * tensor can `Bind` a Buffer, or just has a expression and easy to inline expanded in the consumer's computation.
 *
 * # Reshape
 * An existing tensor can be reshaped with underlying buffer shared.
 * \code
 * auto C1 = C.Reshape({...});
 * \endcode
 */
enum class ViewKind {
  kPrecending = 0,
  kCollapse   = 1,
};

class Tensor : public ir::IrNodeRef {
 public:
  Tensor() = default;
  explicit Tensor(ir::IrNode* n) : IrNodeRef(n) {}

  //! Get number of dimensions.
  inline size_t ndims() const;

  /**
   * Get a tensor with a new shape but underlying buffer shared.
   */
  Tensor Reshape(const std::vector<Expr>& shape);

  /**
   * Slice the preceding n axis and get a new tensor that share the same buffer.
   *
   * \code
   * Tensor A; // shape {M, N, K}
   * Tensor A_slice = A.slice(1);
   * \endcode
   * @param naxis The count of preceding axis to slice.
   * @return a Tensor with its computation inlined.
   */
  Tensor Slice(int naxis);

  /**
   * Take elements from the tensor.
   * This take one or multiple expressions as indices.
   *
   * usage:
   *
   * Tensor A;
   * A(i,j) get the [i][j] element.
   */
  // @{
  Expr operator()(const Expr& a) const { return operator()(std::vector<Expr>({a})); }
  template <typename... Args>
  inline typename std::enable_if<detail::GE(sizeof...(Args), 2), Expr>::type operator()(Args... args) const {
    return operator()({std::forward<Args>(args)...});
  }
  // @}

  /**
   * Take elements from the tensor.
   * @param indices  The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices) const;

  friend bool operator<(const Tensor& a, const Tensor& b);

  //! Expand the inline expression in the body.
  void ExpandInlined();

  _Tensor_* self() { return operator->(); }
  const _Tensor_* self() const { return operator->(); }

  inline const _Tensor_* operator->() const { return As<_Tensor_>(); }
  inline _Tensor_* operator->() { return As<_Tensor_>(); }

  inline operator Expr() const { return Expr(get()); }
};

class ComputeOp;
class PlaceholderOp;
struct ReadCacheRelation;
struct WriteCacheRelation;

//! Store the infomations about some other tensor `compute_at` this tensor.
struct ComputeAtInfo {
  ComputeAtInfo(const std::string& consumer_tensor_name,
                const std::string& producer_tensor_name,
                const std::vector<int>& adjusted_producer_shape,
                int level)
      : consumer_tensor_name(consumer_tensor_name),
        producer_tensor_name(producer_tensor_name),
        adjusted_producer_shape(adjusted_producer_shape),
        level(level) {}

  std::string consumer_tensor_name;
  std::string producer_tensor_name;
  //! The shape of the buffer belong to the producer tensor after compute_at.
  //! NOTE this doesn't support dynamic dimension yet.
  std::vector<int> adjusted_producer_shape;
  //! The preceding offsets for loading the producer, size of this should equal to level+1.
  std::vector<int> preceding_offset_for_producer_load;
  int level;  // NOTE this should be the level of the consumer tensor's transformed range.
};

/**
 * _Tensor_ holds the content of a Tensor.
 *
 * NOTE(All) Some rules:
 *
 * 1. a _Tensor_ is a node in SSA, so every tensor's name should be unique,
 * 2. never try to change a tensor's name, that will cause chaos.
 */
class _Tensor_ : public ExprNode<_Tensor_> {
  //! a pointer to Shared<Stage>, use void* to avoid cyclic definition dependency.
  void* stage_shared{};

 public:
  //! Shape of this tensor(buffer).
  std::vector<Expr> shape;
  //! The domain of each axis(without reduce_axis)
  // TODO(Superjomn) support ISL domain.
  std::vector<Expr> domain;

  std::vector<Var> reduce_axis;
  //! The operation that generates Tensor.
  FunctionRef operation;
  //! Name of this tensor.
  std::string name;
  //! The bound buffer, for each tensor if it is not inline.
  Buffer buffer;

  //! read cache relation if has one.
  std::unique_ptr<ReadCacheRelation> read_cache_relation;
  //! write cache relation if has one.
  std::unique_ptr<WriteCacheRelation> write_cache_relation;

  //! Store the information of all the other producer tensors `compute_at` this tensor.
  std::vector<ComputeAtInfo> compute_at_infos;

  //! Polyhedral element for analysis and schedule.
  poly::Stage* stage();

  std::vector<Expr> domain_with_reduce_axis() const;
  const std::vector<Expr>& domain_without_reduce_axis() const { return domain; }

  //! Generate a tensor from a function.
  static Tensor Make(const std::string& name,
                     Type dtype,
                     const std::vector<Expr>& shape,
                     const std::vector<Expr>& domain,
                     FunctionRef fn,
                     const std::vector<Var>& reduce_axis = {});

  bool compute_inline{false};

  //! Name of the tensors thouse share buffer with `this` tensor.
  std::set<std::string> tensors_to_share_buffer_with;

  //! Reshape a tensor.
  Tensor BufferShared(const std::string& name, const std::vector<Expr>& shape) const;

  //! Tell whether this tensor is inline.
  bool inlined() const;

  //! Tell whether this tensor represents a tuple (consists of one or multiple tensors as output of a extern Call).
  bool is_tuple() const;
  bool is_tuple_get() const;

  Tensor TupleGet(int offset) const;

  /**
   * Get the names of the dependency(read or write) tensors.
   * e.g. A[i] = C[i]*2 + D[i], A's dependency tensors are {C,D}
   */
  std::set<std::string> GetDependTensorNames() const;

  /**
   * \brief Tell whether this tensor's computation relays on a specific statement.
   * @param statement The name of a statement(equivalent to the id of tensor).
   * @return A boolean.
   */
  bool IsDependOnStatement(const std::string& statement);

  /**
   * Get the names of the tensors thouse this tensor depends on.
   */
  std::set<std::string> DependingTensorNames();

  /**
   * Tell whether this tensor has same shape with \p other.
   */
  bool SameShapeWith(const Tensor& other) const;

  //! Operation related.
  // @{
  bool is_compute_node() const;
  bool is_placeholder_node() const;
  bool is_call_node() const;
  bool is_extern_call_node() const;
  bool is_preceding_view_node() const;
  bool is_buffer_shared_node() const;
  const char* operation_type() const;
  ComputeOp* get_compute_op() const;
  PlaceholderOp* get_placeholder_op() const;
  // @}

  //! The expression generate this tensor, will be empty if it is a PlaceHolder.
  Expr body() const;
  Expr* mutable_body();
  //! Get the expression with `store(tensor)` inserted into the body.
  Expr tensor_store_expanded_body();

  Expr inline_expanded(const std::vector<Expr>& indices);

  //! Tell whether contain a reduce axis.
  bool contains_reduce_axis() const { return !reduce_axis.empty(); }

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  /**
   * The normal axis without reducing ones.
   */
  const std::vector<Var>& axis() const {
    CHECK_EQ(axis_.size(), domain_without_reduce_axis().size());
    return axis_;
  }

  /**
   * The axis with the reduce ones.
   */
  std::vector<Var> axis_with_reduce() const {
    auto axis = axis_;
    axis.insert(axis.end(), reduce_axis.begin(), reduce_axis.end());
    return axis;
  }

  /**
   * Get the tensors thouse depend on the same buffer belong to this tensor.
   */
  const std::set<std::string>& buffer_depended_tensor_names() const { return buffer_depended_tensor_names_; }

  static const IrNodeTy _node_type_ = IrNodeTy::_Tensor_;

  _Tensor_() : ExprNode<_Tensor_>(Float(32)) {}

  bool has_expression() const;

  ~_Tensor_();

  //! Create a buffer belong to this tensor.
  void WithBuffer(const Type& type = Void());
  void WithBuffer(const std::string& memory_type, const Type& type = Void());
  //! Bind to a buffer, will persist data to the buffer in runtime.
  void Bind(lang::Buffer& buffer);  // NOLINT
  void Bind(const Buffer& buffer);
  void UnBind(lang::Buffer& buffer);  // NOLINT

  //! Create the polyhedral element for analysis.
  //! It is based on the shape.
  void InitStage();

  //! Free the memory for stage.
  void DropStage();

  void FakeStage();
  bool is_faked() const;

  //! Initialize the axis field after the shape field is assigned.
  void InitAxis();

  //! Extract the tensors of the buffer this writes to. We should schedule this tensor after those tensors, or there
  //! will be read-write conflicts.
  void ExtractBufferDependedTensors();

  isl::set GenerateIslDomain();

  //! The names of the tensors depend the same buffer and should schedule before this.
  std::set<std::string> buffer_depended_tensor_names_;

  //! Normal axis.
  std::vector<Var> axis_;
};

struct ReadCacheRelation {
  //! Name of the cache tensor.
  std::string cache_name;
  //! Names of the reading tensors.
  std::vector<std::string> readers;
};

struct WriteCacheRelation {
  //! Name of the cache tensor.
  std::string cache_name;
};

class _Operation_;
class Operation : public FunctionRef {
 public:
  Operation() = default;
  explicit Operation(IrNode* n) : FunctionRef(n) {}

  inline const _Operation_* operator->() const { return reinterpret_cast<_Operation_*>(get()); }
  inline _Operation_* operator->() { return reinterpret_cast<_Operation_*>(get()); }

  //! Get the i-th output of the operation.
  // Tensor output(size_t i) const;

  std::string name;
};

class _Operation_ : public ir::FunctionBase {
 public:
  //! Optional name of the operation.
  std::string name;
  //! Optional tag of the operation.
  std::string tag;
  //! Additional attributes of the operation.
  std::map<std::string, IrNodeRef> attrs;

  void Accept(IRVisitor* v) const override {}
  const std::string& func_name() const final { return name; }

  //! The function type.
  virtual const char* func_type() const = 0;
};

}  // namespace ir
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::ir::Tensor> {
  inline size_t operator()(const cinn::ir::Tensor& x) {
    // We treat the tensor's name as the unique identifier.
    return std::hash<std::string>()(x->name);
  }
};

}  // namespace std

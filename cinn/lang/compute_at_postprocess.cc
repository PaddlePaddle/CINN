#include "cinn/lang/compute_at_postprocess.h"

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/tensor.h"
#include "cinn/optim/ir_replace.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/poly/compute_at_transform.h"

namespace cinn {
namespace lang {
using ir::ComputeAtInfo;

namespace detail {

/**
 * Process the producer related Store and Load indices.
 */
struct NormalizeProducerDomainMutator : public ir::IRMutator<> {
  std::map<Var, Expr> offsets;
  std::vector<Var> consumer_axis;
  std::string producer_tuple;

  NormalizeProducerDomainMutator(const std::string& producer_tuple, const std::vector<Var>& consumer_axis)
      : producer_tuple(producer_tuple), consumer_axis(consumer_axis) {}

  void operator()(Expr* forloop) { ir::IRMutator<>::Visit(forloop, forloop); }

  //! Add offsets to store, e.g. offset is i->3, the original store expr is a[i,j] = b[i*2,j], the result expression
  //! will be a[i+3,j] = b[(i+3)*2,j]
  void AddOffsetsToStoreExpr(Expr* expr) {
    LOG(INFO) << "*AddOffsetsToStoreExpr: " << *expr;
    CHECK(expr->As<ir::Store>());
    for (auto& offset : offsets) {
      LOG(INFO) << "Add to axis " << offset.first << " with offset " << offset.first << " => +" << offset.second;
      optim::IrReplace(expr, offset.first, Expr(offset.first) + offset.second);
    }
  }

  /* Set the producer axis to zero in Store node, e.g. a store node, a[c0,c1] = ... will be a[0,0]
   *
   * poly_for (i, cinn_max(0, (po0 - 1)), (i <= (po0 + 1)), 1)
   * {
   *   cache[i, po1] = A[i, po1]
   * }
   *
   * will transform to
   *
   * poly_for (i, cinn_max(0, (po0 - 1)), (i <= (po0 + 1)), 1)
   * {
   *   cache[i, 0] = A[i, po1]
   * }
   */
  void SetProducerAxisToZeroInStore(Expr* expr) {
    auto* node = expr->As<ir::Store>();
    CHECK(node);

    VLOG(3) << "SetProducerAxisToZeroInStore: " << *expr;
    for (auto& indice : node->indices) {
      for (auto& consumer_axis : consumer_axis) {
        VLOG(3) << indice << " set producer axis [" << consumer_axis << "] to 0";
        optim::IrReplace(&indice, consumer_axis, common::make_const(0));
      }
    }
  }

  /*
   * Make producer Store indice start from zero.
   *
   * NOTE the axis here should be producer's axis, `i` in the root function comment.
   *
   * poly_for (i, cinn_max(0, (po0 - 1)), (i <= (po0 + 1)), 1)
   * {
   *   cache[i, po1] = A[i, po1]
   * }
   *
   * will transform to
   *
   * poly_for (i, 0, (i + cinn_max(0, (po0 - 1)) <= (po0 + 1)), 1)
   * {
   *   cache[i, po1] = A[i + cinn_max(0, (po0 - 1)), po1]
   * }
   */
  void AddOffsetToAxisInStoreValue(Expr* expr) {
    optim::Simplify(expr);
    LOG(INFO) << "AddOffsetToAxisInStoreValue to:\n" << *expr;

    auto* node = expr->As<ir::Store>();

    auto loads_but_producer = ir::CollectIRNodes(node->value, [&](const Expr* x) {
      return x->As<ir::Load>() && x->As<ir::Load>()->tensor.as_tensor()->name != node->tensor.as_tensor()->name;
    });

    for (auto& item : loads_but_producer) {
      auto* load = item.As<ir::Load>();
      for (auto& indice : load->indices) {
        for (auto& offset : offsets) {
          VLOG(3) << "*Add indice to [" << indice << "] => [" << offset.first << "] with offset [" << offset.second
                  << "]";
          optim::IrReplace(&Reference(&indice), offset.first, Expr(offset.first) + offset.second);
          VLOG(3) << "get: " << indice;
        }
      }
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();

    if (op->tensor.as_tensor()->name == producer_tuple) {
      // AddOffsetsToStoreExpr(expr);

      // replace the producer axis in store indice to zero.
      SetProducerAxisToZeroInStore(expr);

      // replace the consumer axis in value(not producer) to offset.
      AddOffsetToAxisInStoreValue(expr);
    } else {
      ir::IRMutator<>::Visit(op, expr);
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    if (!common::is_zero(op->min)) {
      auto offset             = op->min;
      node->min               = common::make_const(0);
      node->extent            = node->extent - offset;
      offsets[node->loop_var] = offset;
    }
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    if (!common::is_zero(op->init)) {
      auto offset             = op->init;
      offsets[node->iterator] = offset;
      node->init              = common::make_const(0);
      UpdatePolyForConditionWithOffset(&node->condition, node->iterator, offset);
    }
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void UpdatePolyForConditionWithOffset(Expr* cond, Var iter, Expr offset) {
    optim::IrReplace(cond, iter, Expr(iter) + offset);
  }
};

struct ResetProducerLoadIndiceInConsumerMutator : public ir::IRMutator<> {
  const std::string& producer_tensor_name;
  const std::vector<Var>& consumer_axis;
  const ComputeAtInfo& compute_at_info;

  ResetProducerLoadIndiceInConsumerMutator(const std::string& producer_tensor_name,
                                           const std::vector<Var>& consumer_axis,
                                           const ComputeAtInfo& compute_at_info)
      : producer_tensor_name(producer_tensor_name), consumer_axis(consumer_axis), compute_at_info(compute_at_info) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Load* op, Expr* expr) override {
    VLOG(3) << "Consumer modify Load " << *expr << "'s axis for producer [" << producer_tensor_name << "]";
    auto* node = expr->As<ir::Load>();
    if (op->tensor.as_tensor()->name == producer_tensor_name) {
      CHECK_LE(compute_at_info.preceding_offset_for_producer_load.size(), node->indices.size());
      for (auto axis : consumer_axis) {
        for (auto& indice : node->indices) {
          VLOG(3) << "Consumer Load " << indice << " set axis [" << axis << "] to 0";
          optim::IrReplace(&indice, axis, common::make_const(0));
        }
      }

      for (int i = 0; i < compute_at_info.preceding_offset_for_producer_load.size(); i++) {
        node->indices[i] = node->indices[i] + compute_at_info.preceding_offset_for_producer_load[i];
      }
    }
    // Load not recursive, no need to visit it's items.
  }
};

}  // namespace detail

using ir::ComputeAtInfo;
/**
 * Lets define the consumer tensor as C and the producer tensor as P for short.
 * First, find the forloop generating C, keep the forloop levels in a stack.
 * We need to modify the following
 * 1. P's Store indice(change the parameters to zero)
 * 2. P's Store value, change the parameters in Load to consumer's precending axis
 * 3. replace the precending axis of the P's Load to zero in C
 */
struct CorrectComputeAtRelatedIndiceMutator : public ir::IRMutator<> {
  std::string tensor_name;

  explicit CorrectComputeAtRelatedIndiceMutator(const std::string& tensor_name) : tensor_name(tensor_name) {}

  void operator()(Expr* e) { return ir::IRMutator<>::Visit(e, e); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  void Visit(const ir::For* op, Expr* expr) override {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  /**
   * Normalize the producer's domain, make it start from zero. This is essential for shrink the buffer and inference the
   * buffer size.
   *
   * e.g.
   * for (i=p0; i<3+p0; i++) {
   *   p[i]
   * }
   * will be transformed to
   * for (i=0; i<3; i++) {
   *   p[i+p0]
   * }
   *
   * @param producer_forloop_root The root of the producer's own axis, not the axis of consumer.
   *
   * About the \p producer_forloop_root, after compute_at schedule,
   * // consumer iter ci
   * for (ci) {
   *   // producer iter pi
   *   for (pi) {
   *   }
   * }
   * The pi should be the \p producer_forloop_root
   */
  void NormalizeProducerDomain(Expr* producer_forloop_root,
                               const std::string& producer_tuple,
                               const std::vector<Var>& consumer_axis) {
    VLOG(4) << "Normalize producer domain: " << producer_tuple;
    VLOG(4) << "producer_forloop_root:\n" << *producer_forloop_root;
    VLOG(4) << "consumer_axis:";
    for (auto& var : consumer_axis) {
      VLOG(4) << "iter: " << var;
    }

    detail::NormalizeProducerDomainMutator(producer_tuple, consumer_axis)(producer_forloop_root);
  }

  //! Reset the indice of the producer Load in Consumer.
  // Here we just set the minimum consumer axis to zero. e.g., for consumer statement such as
  // `C[i] = A[i-1]+A[i]+A[i+1]` and level set to 0, the result statement will be `C[i] = A[0]+A[1]+A[2]`, this includes
  // the following steps:
  // 1. make the preceding level+1 axis to zero in producer load, we get `C[i] = A[-1]+A[0]+A[1]`.
  // 2. for each adjusted axis, add an offset stored in ComputeAtInfo to make the minimum indice zero, then we get `C[i]
  // = A[0]+A[1]+A[2]`.
  void ResetProducerLoadIndiceInConsumer(const std::vector<Var>& consumer_axis,
                                         Expr* consumer_store_expr,
                                         const std::string& producer_tensor_name,
                                         const ComputeAtInfo& compute_at_info) {
    detail::ResetProducerLoadIndiceInConsumerMutator(
        producer_tensor_name, consumer_axis, compute_at_info)(consumer_store_expr);
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();

    if (op->tensor.as_tensor()->name != tensor_name) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }

    // get the target consumer
    auto& compute_at_infos = op->tensor.as_tensor()->compute_at_infos;
    CHECK(!compute_at_infos.empty());

    std::vector<Var> levels;
    for (Expr* forloop : forloop_stack) {
      auto* for_n      = forloop->As<ir::For>();
      auto* poly_for_n = forloop->As<ir::PolyFor>();
      if (for_n)
        levels.push_back(for_n->loop_var);
      else if (poly_for_n)
        levels.push_back(poly_for_n->iterator);
      else
        NOT_IMPLEMENTED
    }

    for (auto& compute_at_info : compute_at_infos) {
      VLOG(4) << "compute_at: " << compute_at_info.producer_tensor_name;
      detail::ReplaceParamWithConsumerAxis(compute_at_info, levels, forloop_stack.front());
    }

    for (auto& compute_at_info : compute_at_infos) {
      int level = compute_at_info.level;
      std::vector<Var> consumer_aixs(levels.begin(), levels.begin() + level + 1);
      Expr* producer_forloop_root;
      if (forloop_stack[level]->As<ir::For>()) {
        producer_forloop_root = &forloop_stack[level]->As<ir::For>()->body;
      } else {
        producer_forloop_root = &forloop_stack[level]->As<ir::PolyFor>()->body;
      }

      auto forloop_stack_to_store = GetForloopStackToStore(producer_forloop_root, compute_at_info.producer_tensor_name);
      producer_forloop_root = forloop_stack_to_store.empty() ? forloop_stack[level] : forloop_stack_to_store.front();
      NormalizeProducerDomain(producer_forloop_root, compute_at_info.producer_tensor_name, consumer_aixs);
      ResetProducerLoadIndiceInConsumer(
          consumer_aixs, forloop_stack[level], compute_at_info.producer_tensor_name, compute_at_info);
    }
  }

  std::vector<Expr*> forloop_stack;
};

void ProcessComputeAtInfo(Expr* expr) {
  // 1. collect all the consumer tensors thouse have compute_at_infos.
  // 2. for each producer tensor, reset the producer tensor loads indice.

  // first, visit the consumer tensor with compute_at info.
  // second, in the forloop stack, find the producer tensor
  //    - set the presending axis to zero in producer's Store node and Load node
  //    - replace the ISL parameter to the precending axis
  // in consumer, reset presending axis in producer's Load to zero.

  auto tensor_with_compute_at_infos = ir::CollectIRNodes(
      *expr, [&](const Expr* x) { return x->as_tensor() && !x->as_tensor()->compute_at_infos.empty(); });

  for (auto& tensor : tensor_with_compute_at_infos) {
    VLOG(4) << "consumer: " << tensor;
    CorrectComputeAtRelatedIndiceMutator(tensor.as_tensor()->name)(expr);
  }
}

void UpdateComputeAtBufferShape(Expr* expr) {
  auto tensor_with_compute_at_infos = ir::CollectIRNodes(*expr, [&](const Expr* x) {
    return x->as_tensor() && !x->as_tensor()->inlined() && !x->as_tensor()->compute_at_infos.empty();
  });

  auto tensor_map = ir::CollectTensorMap(
      *expr, [&](const Expr* x) { return !x->as_tensor()->inlined() && x->as_tensor()->buffer.defined(); });

  std::unordered_map<std::string, ir::ComputeAtInfo*> buffer_to_compute_at_info;
  for (auto& item : tensor_map) {
    auto& compute_at_infos = item.second.as_tensor()->compute_at_infos;
    if (compute_at_infos.empty()) continue;
    for (auto& compute_at : compute_at_infos) {
      auto& producer_tensor = tensor_map.at(compute_at.producer_tensor_name);
      buffer_to_compute_at_info[producer_tensor.as_tensor()->buffer->name] = &compute_at_infos.front();
    }
  }

  auto process_tensor = [&](ir::_Tensor_* tensor, const ComputeAtInfo& compute_at_info) {
    tensor->shape.clear();
    for (int v : compute_at_info.adjusted_producer_shape) {
      tensor->shape.push_back(Expr(v));
    }
    VLOG(4) << "Updated tensor: " << ir::Tensor(tensor);
  };

  auto process_buffer = [&](ir::_Buffer_* buffer, const ComputeAtInfo& compute_at_info) {
    buffer->shape.clear();
    for (int v : compute_at_info.adjusted_producer_shape) {
      buffer->shape.push_back(Expr(v));
    }
    VLOG(4) << "Updated buffer: " << ir::Buffer(buffer);
  };

  auto process_alloca = [&](ir::Alloc* alloca, const ComputeAtInfo& compute_at_info) {
    alloca->extents.clear();
    for (int v : compute_at_info.adjusted_producer_shape) {
      alloca->extents.push_back(Expr(v));
    }
    VLOG(4) << "Updated alloca: " << Expr(alloca);
  };

  auto tensors = ir::CollectIRNodes(*expr, [&](const Expr* x) { return x->as_tensor() && !x->as_tensor()->inlined(); });
  for (auto& t : tensors) {
    if (!t.as_tensor()->buffer.defined() || !buffer_to_compute_at_info.count(t.as_tensor()->buffer->name)) continue;
    auto& buffer       = t.as_tensor()->buffer;
    auto compute_at_it = buffer_to_compute_at_info.find(buffer->name);
    if (compute_at_it != buffer_to_compute_at_info.end()) {
      process_tensor(&Reference(t.as_tensor()), *compute_at_it->second);
      process_buffer(Reference(t.as_tensor()).buffer->self(), *compute_at_it->second);
      VLOG(4) << "resizing buffer " << t;
      VLOG(4) << "resizing tensor " << t.as_tensor()->buffer;
    }
  }

  // update lowered func temporay buffers
  auto lowered_fns = ir::CollectIRNodes(*expr, [&](const Expr* x) { return x->as_lowered_func(); });
  for (auto& lowered_fn : lowered_fns) {
    auto* node = lowered_fn.as_lowered_func();
    for (auto& buf : node->temp_bufs) {
      auto compute_at_it = buffer_to_compute_at_info.find(buf->name);
      if (compute_at_it != buffer_to_compute_at_info.end()) {
        process_buffer(Reference(&buf).operator->(), *compute_at_it->second);
      }
    }
  }
}

namespace detail {

/**
 *
 * e.g. The original code is as follows:
 *
 *  poly_for (po0, 0, (po0 <= 9), 1)
 *  {
 *    poly_for (po1, 0, (po1 <= 9), 1)
 *    {
 *      {
 *        if (((((_cp_C_0 >= 0) and (_cp_C_0 <= 9)) and (_cp_C_1 >= 0)) and (_cp_C_1 <= 9))) {
 *          poly_for (i, cinn_max(0, (_cp_C_0 - 1)), (i <= (_cp_C_0 + 1)), 1)
 *          {
 *            cache(i, _cp_C_1)
 *          }
 *        }
 *        C(po0, po1)
 *      }
 *    }
 *  }
 * Note that, the _cp_C_0 like variables are ISL parameters.
 *
 * will transform to
 *
 * poly_for (po0, 0, (po0 <= 9), 1)
 *  {
 *   poly_for (po1, 0, (po1 <= 9), 1)
 *   {
 *     {
 *       if (((((po0 >= 0) and (po0 <= 9)) and (po1 >= 0)) and (po1 <= 9))) {
 *         poly_for (i, cinn_max(0, (po0 - 1)), (i <= (po0 + 1)), 1)
 *         {
 *           cache[i, po1] = A[i, po1]
 *         }
 *       }
 *       C[po0, po1] = select((po0 < 10), (((cache[(po0 - 1), po1] + cache[po0, po1]) + cache[(po0 + 1), po1]) + B[po0,
 * po1]), 0)
 *      }
 *    }
 *  }
 *
 * @param info The compute at information.
 * @param axis The consumer axis.
 * @param consumer_forloop_root The first level of forloop of consumer.
 */
void ReplaceParamWithConsumerAxis(const ComputeAtInfo& info,
                                  const std::vector<Var>& axis,
                                  Expr* consumer_forloop_root) {
  CHECK_LE(info.level + 1, axis.size());
  // replace the params to consumer's precending level+1 axis.
  for (int i = 0; i < info.level + 1; i++) {
    Var var(poly::GenConsumerParamName(info.consumer_tensor_name.c_str(), i));
    VLOG(4) << "replacing " << var << " to " << axis[i];
    optim::IrReplace(consumer_forloop_root, Expr(var), axis[i]);
  }

  VLOG(4) << "After ReplaceParamWithConsumerAxis:\n" << *consumer_forloop_root;
}

}  // namespace detail

}  // namespace lang
}  // namespace cinn

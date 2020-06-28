#include "cinn/lang/lower_impl.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace lang {
namespace detail {

#define CREATE_GNODE(k__) auto* n##k__ = graph->RetriveNode(#k__);

TEST(CreateCompGraph, single_layer) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  LOG(INFO) << C->expr_fields().size();
  for (auto* e : C->expr_fields()) {
    LOG(INFO) << "e: " << *e;
  }

  auto graph = CreateCompGraph({A, B, C});

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /* generated graph
    digraph G {
       node_0[label="A"]
       node_1[label="B"]
       node_2[label="C"]
       node_0->node_2
       node_1->node_2
    } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)

  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nC));
}

TEST(CreateCompGraph, multi_layers) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); }, "E");

  auto graph = CreateCompGraph({A, B, E});

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
   digraph G {
     node_0[label="A"]
     node_1[label="B"]
     node_3[label="C"]
     node_4[label="D"]
     node_2[label="E"]
     node_0->node_2
     node_0->node_3
     node_1->node_2
     node_1->node_4
     node_1->node_3
     node_3->node_2
     node_3->node_4
     node_4->node_2
   } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)
  CREATE_GNODE(D)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 5);

  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nC));

  ASSERT_TRUE(nC->IsLinkedTo(nD));
  ASSERT_TRUE(nB->IsLinkedTo(nD));

  ASSERT_TRUE(nA->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
  ASSERT_TRUE(nC->IsLinkedTo(nE));
  ASSERT_TRUE(nD->IsLinkedTo(nE));
}

TEST(CreateCompGraph, inline_compatible) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); }, "E");

  D->stage()->ComputeInline();

  auto graph = CreateCompGraph({A, B, E}, true);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
    digraph G {
    node_0[label="A"]
    node_1[label="B"]
    node_3[label="C"]
    node_2[label="E"]
    node_0->node_2
    node_0->node_3
    node_1->node_2
    node_1->node_3
    node_3->node_2
    } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 4);
  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nA->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
}

TEST(CreateCompGraph, inline_compatible1) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); }, "E");

  C->stage()->ComputeInline();

  auto graph = CreateCompGraph({A, B, E}, true);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
  digraph G {
     node_0[label="A"]
     node_1[label="B"]
     node_3[label="D"]
     node_2[label="E"]
     node_0->node_2
     node_1->node_2
     node_1->node_3
     node_3->node_2
  } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(D)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 4);

  ASSERT_TRUE(nA->IsLinkedTo(nE));
  ASSERT_TRUE(nD->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nD));
  ASSERT_TRUE(nD->IsLinkedTo(nE));
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn

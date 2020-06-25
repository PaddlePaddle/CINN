#include "cinn/poly/stage.h"

#include <gtest/gtest.h>

#include <set>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, {}, ir::CallType::CINN, ir::FunctionRef(), 0);
  return expr;
}

TEST(Stage, split) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  auto ele            = Stage::New(domain);
  auto [outer, inner] = ele->Split(Iterator("i"), 4);  // NOLINT
  LOG(INFO) << ele->transform();
  EXPECT_EQ(utils::GetStreamCnt(ele->transform()),
            "{ S[i, j] -> S[i_outer, i_inner, j' = j] : (-i + i_inner) mod 4 = 0 and -3 + i <= 4i_outer <= i and 0 <= "
            "i_inner <= 3 }");

  EXPECT_EQ(outer.id, "i_outer");
  EXPECT_EQ(inner.id, "i_inner");
}

TEST(Stage, tile) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);

  auto [outer0, inner0, outer1, inner1] = ele->Tile(Iterator("i"), Iterator("j"), 4, 6);  // NOLINT
  LOG(INFO) << ele->transform();
  EXPECT_EQ(outer0.id, "i_outer");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(inner0.id, "i_inner");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(
      utils::GetStreamCnt(ele->transform()),
      "{ S[i, j, k] -> S[i_outer, i_inner, j_outer, j_inner, k' = k] : (-i + i_inner) mod 4 = 0 and (-j + j_inner) mod "
      "6 = 0 and -3 + i <= 4i_outer <= i and 0 <= i_inner <= 3 and -5 + j <= 6j_outer <= j and 0 <= j_inner <= 5 }");
}

TEST(Stage, reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{i, k, j}});
  LOG(INFO) << ele->transform();
}

TEST(Stage, split_reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele            = Stage::New(domain);
  auto [outer, inner] = ele->Split(Iterator("i"), 4);  // NOLINT

  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{outer, k, inner, j}});
  LOG(INFO) << ele->transform();
}

TEST(ComputeAtRelation, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain0(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  isl::set domain1(ctx, "{ D[a,b,c,d]: 0<=a,b,c,d<=100 }");

  auto stage0 = Stage::New(domain0);
  auto stage1 = Stage::New(domain0);

  ComputeAtRelation relation;
  relation.stage = stage1;
  relation.level = 2;
  ASSERT_TRUE(relation.IsCompatible(stage0.get()));
}

TEST(Stage, Fuse) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele            = Stage::New(domain);
  auto [outer, inner] = ele->Split(Iterator("i"), 4);  // NOLINT
  LOG(INFO) << "split: " << ele->transform();
  ele->Fuse(outer, inner);
  LOG(INFO) << "fused: " << ele->transform();
}

TEST(Stage, Fuse1) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator i("i");
  Iterator j("j");
  auto n = ele->Fuse(i, j);
  LOG(INFO) << "fused: " << ele->transform();
}

TEST(ComputeAt, Before) {
  Expr M(100), N(200);

  Placeholder<float> A("A", {M, N});

  auto create_module = [&] {
    // cached compute way
    auto cache_prepare = Compute({M, N} /*domain*/, [&](Var i, Var j) { return A(i, j); }, "cache", {}, {N} /*shape*/);

    auto transformed_compute = Compute(
        {M, N}, [&](Var i, Var j) { return Expr(1.f); }, "transformed");

    return std::make_tuple(cache_prepare, transformed_compute);
  };

  {  // C_init Before C
    auto [cache_prepare, transformed_compute] = create_module();

    cache_prepare->stage()->ComputeAt(transformed_compute->stage(), 1, Stage::kComputeAtBefore);

    // codegen and compare
    auto fn = Lower("fn", {A, cache_prepare, transformed_compute});

    auto target = utils::Trim(R"ROC(
function fn (_A, _cache, _transformed)
{
  for (i, 100)
  {
    for (j, 200)
    {
      cache[i] = A[i, j]
      transformed[i, j] = 1
    }
  }
}
)ROC");

    ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), target);
  }
  {  // C_init After C
    auto [cache_prepare, transformed_compute] = create_module();

    cache_prepare->stage()->ComputeAt(transformed_compute->stage(), 1, Stage::kComputeAtAfter);

    // codegen and compare
    auto fn = Lower("fn", {A, cache_prepare, transformed_compute});

    auto target = utils::Trim(R"ROC(
function fn (_A, _cache, _transformed)
{
  for (i, 100)
  {
    for (j, 200)
    {
      transformed[i, j] = 1
      cache[i] = A[i, j]
    }
  }
}
)ROC");

    ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), target);
  }
}

TEST(ComputeAt, After) {
  Expr M(100), N(200);

  Placeholder<float> A("A", {M, N});

  // cached compute way
  auto cache_prepare = Compute({M, N} /*domain*/, [&](Var i, Var j) { return A(i, j); }, "cache", {}, {N} /*shape*/);

  auto transformed_compute = Compute(
      {M, N}, [&](Var i, Var j) { return cache_prepare(j); }, "transformed");

  cache_prepare->stage()->ComputeAt(transformed_compute->stage(), 1);

  // codegen and compare
  auto fn = Lower("fn", {A, cache_prepare, transformed_compute});

  auto target = utils::Trim(R"ROC(
function fn (_A, _cache, _transformed)
{
  for (i, 100)
  {
    for (j, 200)
    {
      cache[i] = A[i, j]
      transformed[i, j] = cache[j]
    }
  }
}
)ROC");

  ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), target);
}

TEST(ComputeInline, basic) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");
  auto B2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) + 1.f; }, "B2");

  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B2(i, j) * 2.f; }, "C");

  B->stage()->ComputeInline();
  B1->stage()->ComputeInline();
  B2->stage()->ComputeInline();

  auto inlined_B = B->inline_expanded({Expr(2), Expr(1)});
  ASSERT_EQ("(A[2, 1] + 1)", utils::GetStreamCnt(inlined_B));

  auto fn = Lower("fn", {A, C});

  LOG(INFO) << "fn:\n" << fn;

  auto target = R"ROC(
function fn (_A, _C)
{
  for (i, 100)
  {
    for (j, 200)
    {
      C[i, j] = (6 + (2 * A[i, j]))
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::Trim(utils::GetStreamCnt(fn)));
}

TEST(ComputeInline, complex_graph) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");
  auto B2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) + 1.f; }, "B2");

  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) * 2.f; }, "C");
  auto C1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) * 2.f; }, "C1");
  auto C2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B2(i, j) * 2.f; }, "C2");

  B->stage()->ComputeInline();
  B1->stage()->ComputeInline();
  B2->stage()->ComputeInline();

  auto fn = Lower("fn", {A, C, C1, C2});

  LOG(INFO) << "fn:\n" << fn;

  auto target = R"ROC(
function fn (_A, _C, _C1, _C2)
{
  for (i, 100)
  {
    for (j, 200)
    {
      C2[i, j] = (6 + (2 * A[i, j]))
    }
  }
  for (i, 100)
  {
    for (j, 200)
    {
      C1[i, j] = (4 + (2 * A[i, j]))
    }
  }
  for (i, 100)
  {
    for (j, 200)
    {
      C[i, j] = (2 + (2 * A[i, j]))
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::Trim(utils::GetStreamCnt(fn)));
}

TEST(ShareBufferWith, basic) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");

  B1->stage()->ShareBufferWith(B);

  auto fn = Lower("fn", {A, B, B1});

  LOG(INFO) << "fn:\n" << fn;

  Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  CodeGenC codegen(common::DefaultHostTarget());
  codegen.SetInlineBuiltinCodes(false);

  LOG(INFO) << "\n" << codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
}

}  // namespace poly
}  // namespace cinn

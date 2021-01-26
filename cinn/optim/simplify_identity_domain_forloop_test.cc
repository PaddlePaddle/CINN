#include "cinn/optim/simplify_identity_domain_forloop.h"
#include <gtest/gtest.h>
#include "cinn/cinn.h"

namespace cinn {
namespace optim {

TEST(SimplifyIdentityDomainForloop, basic) {
  Var i("i");

  using namespace ir;  // NOLINT

  Expr M(512);
  Placeholder<float> A("A", {M});

  Tensor C = Compute({M}, [&](Var i) { return A(i) + 1.f; });

  auto forloop  = ir::For::Make(i, Expr(0), Expr(1), ir::ForType::Serial, ir::DeviceAPI::Host, C(i));
  auto forloop1 = ir::For::Make(Var("j"), Expr(0), Expr(1), ir::ForType::Serial, ir::DeviceAPI::Host, forloop);
  auto forloop2 = ir::For::Make(Var("k"), Expr(0), Expr(100), ir::ForType::Serial, ir::DeviceAPI::Host, forloop1);

  LOG(INFO) << "for:\n" << forloop2 << std::endl;

  SimplifyIdentityDomainForloop(&forloop2);
  LOG(INFO) << "for:\n" << forloop2 << std::endl;

  ASSERT_EQ(utils::GetStreamCnt(forloop2), utils::Trim(R"ROC(
for (k, 0, 100)
tensor[0]
)ROC"));
}

}  // namespace optim
}  // namespace cinn
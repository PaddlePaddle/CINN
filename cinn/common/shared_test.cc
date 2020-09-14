#include "cinn/common/shared.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/common/object.h"

namespace cinn {
namespace common {

struct A : public Object {
  const char *type_info() const override { return "A"; }

  Shared<A> other;
};

class B : public Object {};

TEST(Shared, test) {
  Shared<A> a_ref(make_shared<A>());
  ASSERT_EQ(ref_count(a_ref.get()).val(), 1);

  {  // local copy
    Shared<A> b = a_ref;
    EXPECT_EQ(ref_count(a_ref.get()).val(), 2);
    ASSERT_EQ(ref_count(b.get()).val(), 2);
  }

  ASSERT_EQ(ref_count(a_ref.get()).val(), 1);
}

TEST(Shared, cycle_share) {
  {
    Shared<A> a_ref(make_shared<A>());
    a_ref->other = a_ref;
    ASSERT_EQ(a_ref->__ref_count__.val(), 2);
  }
}

}  // namespace common
}  // namespace cinn

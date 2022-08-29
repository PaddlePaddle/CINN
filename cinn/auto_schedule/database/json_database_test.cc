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

#include "cinn/auto_schedule/database/json_database.h"

#include <gtest/gtest.h>

#include <vector>

#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

void AddTestRecords(JSONDatabase& test_db) {
  test_db.AddRecord(TuningRecord({"k1", 1.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k2", 2.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k2", 3.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k3", 3.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k3", 4.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k3", 5.0, SearchState(ir::ModuleExpr())}));
  test_db.AddRecord(TuningRecord({"k4", 4.0, SearchState(ir::ModuleExpr())}));

  SearchState state1(std::move(ir::ModuleExpr()));
  SearchState state2(std::move(ir::ModuleExpr()));
  state1.predicted_cost = 1.2;
  state2.predicted_cost = 1.0;
  test_db.AddRecord(TuningRecord({"k4", 2.0, state1}));
  test_db.AddRecord(TuningRecord({"k4", 3.0, state2}));
}

class TestJSONDatabase : public ::testing::Test {
 public:
  TestJSONDatabase() : test_db(2, "./test_record.json", true) {
    if (0 == test_db.Size()) {
      AddTestRecords(test_db);
    }
  }

  void SetUp() override {}
  JSONDatabase test_db;
};

TEST_F(TestJSONDatabase, Basic) {
  ASSERT_EQ(test_db.Size(), 7);
  auto records = test_db.LookUp("k3");
  // check the max number of stored candidates will
  // be restricted to capacity_per_task
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);
}

TEST_F(TestJSONDatabase, ReInit) {
  test_db.ReInit("./test_record.json", false);
  ASSERT_EQ(test_db.Size(), 7);
  auto records = test_db.LookUp("k3");
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);

  test_db.ReInit("./new_test_record.json", true);
  ASSERT_EQ(test_db.Size(), 0);
  records = test_db.LookUp("k1");
  ASSERT_EQ(test_db.Count("k1"), 0);
  ASSERT_EQ(records.size(), 0);
}

TEST_F(TestJSONDatabase, GetTopK) {
  ASSERT_TRUE(test_db.GetTopK("k5", 2).empty());

  auto states = test_db.GetTopK("k4", 3);
  ASSERT_EQ(states.size(), 2);
  // Because the save / load of SearchState has not been added,
  // ths predicted_cost is temporarily - 1
  EXPECT_FLOAT_EQ(states[0].predicted_cost, -1);
  EXPECT_FLOAT_EQ(states[1].predicted_cost, -1);
}

}  // namespace auto_schedule
}  // namespace cinn

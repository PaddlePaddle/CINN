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

#include "cinn/auto_schedule/database/jsonfile_database.h"

#include <gtest/gtest.h>

#include <vector>

#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

void AddTestRecords(JSONFileDatabase& test_db) {
  test_db.AddRecord(TuningRecord("k1", 1.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k2", 2.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k2", 3.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k3", 3.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k3", 4.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k3", 5.0, SearchState(ir::ModuleExpr())));
  test_db.AddRecord(TuningRecord("k4", 4.0, SearchState(ir::ModuleExpr())));

  SearchState state1(std::move(ir::ModuleExpr()));
  SearchState state2(std::move(ir::ModuleExpr()));
  state1.predicted_cost = 1.2;
  state2.predicted_cost = 1.0;
  test_db.AddRecord(TuningRecord("k4", 2.0, state1));
  test_db.AddRecord(TuningRecord("k4", 3.0, state2));
}

class TestJSONFileDatabase : public ::testing::Test {
 public:
  TestJSONFileDatabase() : test_db(2, "./test_record.json", true) {
    if (0 == test_db.Size()) {
      AddTestRecords(test_db);
    }
  }

  void SetUp() override {}
  JSONFileDatabase test_db;
};

TEST_F(TestJSONFileDatabase, SerializeAndDeserialize) {
  TuningRecord record1("test", 1.0, SearchState(ir::ModuleExpr()));
  std::string str = test_db.RecordToJSON(record1);
  EXPECT_EQ(str, "{\"taskKey\":\"test\",\"executionCost\":1}");

  TuningRecord record2 = test_db.JSONToRecord(str);
  EXPECT_EQ(record1.task_key, record2.task_key);
  EXPECT_EQ(record1.execution_cost, record2.execution_cost);
}

TEST_F(TestJSONFileDatabase, SaveLoad) {
  std::vector<std::string> strs = ReadLinesFromFile("./test_record.json");
  ASSERT_EQ(strs.size(), 9);
  EXPECT_EQ(strs[0], "{\"taskKey\":\"k1\",\"executionCost\":1}");
  EXPECT_EQ(strs[1], "{\"taskKey\":\"k2\",\"executionCost\":2}");
  EXPECT_EQ(strs[2], "{\"taskKey\":\"k2\",\"executionCost\":3}");
  EXPECT_EQ(strs[3], "{\"taskKey\":\"k3\",\"executionCost\":3}");
  EXPECT_EQ(strs[4], "{\"taskKey\":\"k3\",\"executionCost\":4}");
  EXPECT_EQ(strs[5], "{\"taskKey\":\"k3\",\"executionCost\":5}");
  EXPECT_EQ(strs[6], "{\"taskKey\":\"k4\",\"executionCost\":4}");
  EXPECT_EQ(strs[7], "{\"taskKey\":\"k4\",\"executionCost\":2}");
  EXPECT_EQ(strs[8], "{\"taskKey\":\"k4\",\"executionCost\":3}");
}

TEST_F(TestJSONFileDatabase, Basic) {
  ASSERT_EQ(test_db.Size(), 7);
  auto records = test_db.LookUp("k3");
  // check the max number of stored candidates will
  // be restricted to capacity_per_task
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);
}

TEST_F(TestJSONFileDatabase, GetTopK) {
  ASSERT_TRUE(test_db.GetTopK("k5", 2).empty());

  auto states = test_db.GetTopK("k4", 3);
  ASSERT_EQ(states.size(), 2);
  // Because the save/load of SearchState has not been added,
  // the predicted_cost is temporarily -1
  EXPECT_FLOAT_EQ(states[0].predicted_cost, -1);
  EXPECT_FLOAT_EQ(states[1].predicted_cost, -1);
}

}  // namespace auto_schedule
}  // namespace cinn

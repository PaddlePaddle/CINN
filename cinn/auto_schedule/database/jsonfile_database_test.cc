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

#include <fstream>
#include <vector>

#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/auto_schedule/task/task_registry.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

// Return lowerd ir AST for example functions used in this test
std::vector<ir::LoweredFunc> LowerCompute(const std::vector<int>& shape, const Target& target) {
  CHECK(shape.size() == 2) << "shape should be 2";
  std::vector<Expr> domain;
  for (auto i = 0; i < shape.size(); ++i) {
    domain.emplace_back(shape[i]);
  }

  Placeholder<float> A("A", domain);
  ir::Tensor B, C;

  B = Compute(
      domain, [&A](Var i, Var j) { return A(i, j); }, "B");
  C = Compute(
      domain, [&B](Var i, Var j) { return B(i, j); }, "C");

  return cinn::lang::LowerVec("test_func", CreateStages({A, B}), {A, B}, {}, {}, nullptr, target, true);
}

// Create a new IRSchedule with copied ir::LoweredFunc AST
ir::IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs, const std::string& task_key) {
  std::vector<Expr> exprs;
  for (auto&& func : lowered_funcs) {
    exprs.emplace_back(optim::IRCopy(func->body));
  }
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  task_registry->Regist(task_key, ir::ModuleExpr(exprs));

  return ir::IRSchedule(ir::ModuleExpr(exprs));
}

class TestJSONFileDatabase : public ::testing::Test {
 public:
  TestJSONFileDatabase() : record_file_path("/tmp/test_record.json"), test_db(2, record_file_path, true) {}

  void SetUp() override { lowered_funcs = LowerCompute({32, 32}, target); }

  void TearDown() override {
    auto isFileExists = [](const std::string& file_path) -> bool {
      std::ifstream f(file_path.c_str());
      return f.good();
    };
    if (isFileExists(record_file_path)) {
      if (remove(record_file_path.c_str()) == 0) {
        LOG(INFO) << "Successfully deleted file: " << record_file_path;
      } else {
        LOG(INFO) << "failed to delete file: " << record_file_path;
      }
    } else {
      LOG(INFO) << "file: " << record_file_path << "does not exist.";
    }
  }

  std::string record_file_path;
  JSONFileDatabase test_db;
  std::vector<ir::LoweredFunc> lowered_funcs;
  Target target = common::DefaultHostTarget();
};

TEST_F(TestJSONFileDatabase, Serialize) {
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs, "test");
  auto fused            = ir_sch.Fuse("B", {0, 1});
  VLOG(3) << "after Fuse, Expr: " << fused;

  TuningRecord record1("test", 1.0, 2.0, std::move(ir_sch));
  std::string str = test_db.RecordToJSON(record1);
  VLOG(3) << "RecordToJSON: " << str;
  // Because the serialization of protobuf does not guarantee the order, we give all possible results.
  std::string case1 =
      "{\"taskKey\":\"test\",\"executionCost\":1,\"predictedCost\":2,\"trace\":{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"loops_index\",\"dtype\":\"INTS\",\"ints\":[0,1]},{\"name\":\"block_"
      "name\",\"dtype\":\"STRING\",\"s\":\"B\"}]}]}}";
  std::string case2 =
      "{\"taskKey\":\"test\",\"executionCost\":1,\"predictedCost\":2,\"trace\":{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"block_name\",\"dtype\":\"STRING\",\"s\":\"B\"},{\"name\":\"loops_"
      "index\",\"dtype\":\"INTS\",\"ints\":[0,1]}]}]}}";
  EXPECT_EQ(true, str == case1 || str == case2);
}

TEST_F(TestJSONFileDatabase, SaveLoad) {
  ir::IRSchedule ir_sch1 = MakeIRSchedule(lowered_funcs, "k1");
  auto fused1            = ir_sch1.Fuse("B", {0, 1});
  ir::IRSchedule ir_sch2 = MakeIRSchedule(lowered_funcs, "k2");

  test_db.AddRecord(TuningRecord("k1", 1.0, 1.5, std::move(ir_sch1)));
  test_db.AddRecord(TuningRecord("k2", 3.0, 3.5, std::move(ir_sch2)));

  std::vector<std::string> strs = ReadLinesFromFile(record_file_path);
  ASSERT_EQ(strs.size(), 2);
  // Because the serialization of protobuf does not guarantee the order, we give all possible results.
  std::string case1 =
      "{\"taskKey\":\"k1\",\"executionCost\":1,\"predictedCost\":1.5,\"trace\":{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"loops_index\",\"dtype\":\"INTS\",\"ints\":[0,1]},{\"name\":\"block_"
      "name\",\"dtype\":\"STRING\",\"s\":\"B\"}]}]}}";
  std::string case2 =
      "{\"taskKey\":\"k1\",\"executionCost\":1,\"predictedCost\":1.5,\"trace\":{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"block_name\",\"dtype\":\"STRING\",\"s\":\"B\"},{\"name\":\"loops_"
      "index\",\"dtype\":\"INTS\",\"ints\":[0,1]}]}]}}";
  EXPECT_EQ(true, strs[0] == case1 || strs[0] == case2);
  EXPECT_EQ(strs[1], "{\"taskKey\":\"k2\",\"executionCost\":3,\"predictedCost\":3.5,\"trace\":{}}");
}

TEST_F(TestJSONFileDatabase, Basic) {
  test_db.AddRecord(TuningRecord("k1", 1.0, 1.0, MakeIRSchedule(lowered_funcs, "k1")));
  test_db.AddRecord(TuningRecord("k2", 2.0, 1.0, MakeIRSchedule(lowered_funcs, "k2")));
  test_db.AddRecord(TuningRecord("k2", 3.0, 1.0, MakeIRSchedule(lowered_funcs, "k2")));
  test_db.AddRecord(TuningRecord("k3", 3.0, 8.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k3", 4.0, 7.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k3", 5.0, 6.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k4", 4.0, 1.0, MakeIRSchedule(lowered_funcs, "k4")));

  ASSERT_EQ(test_db.Size(), 6);
  auto records = test_db.LookUp("k3");
  // check the max number of stored candidates will
  // be restricted to capacity_per_task
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);

  JSONFileDatabase new_db(2, record_file_path, false);
  ASSERT_EQ(test_db.Size(), 6);
  auto new_records = test_db.LookUp("k3");
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);
}

TEST_F(TestJSONFileDatabase, GetTopK) {
  test_db.AddRecord(TuningRecord("k1", 1.0, 1.0, MakeIRSchedule(lowered_funcs, "k1")));
  test_db.AddRecord(TuningRecord("k2", 2.0, 1.0, MakeIRSchedule(lowered_funcs, "k2")));
  test_db.AddRecord(TuningRecord("k2", 3.0, 1.0, MakeIRSchedule(lowered_funcs, "k2")));
  test_db.AddRecord(TuningRecord("k3", 3.0, 1.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k3", 4.0, 1.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k3", 5.0, 1.0, MakeIRSchedule(lowered_funcs, "k3")));
  test_db.AddRecord(TuningRecord("k4", 4.0, 2.0, MakeIRSchedule(lowered_funcs, "k4")));
  test_db.AddRecord(TuningRecord("k4", 2.0, 1.2, MakeIRSchedule(lowered_funcs, "k4")));
  test_db.AddRecord(TuningRecord("k4", 3.0, 1.0, MakeIRSchedule(lowered_funcs, "k4")));

  auto states = test_db.GetTopK("k4", 3);
  ASSERT_EQ(states.size(), 2);
  EXPECT_FLOAT_EQ(states[0].predicted_cost, 1.2);
  EXPECT_FLOAT_EQ(states[1].predicted_cost, 1.0);
}

}  // namespace auto_schedule
}  // namespace cinn

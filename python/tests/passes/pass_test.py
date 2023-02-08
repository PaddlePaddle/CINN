# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from cinn.frontend import *
from cinn.common import *
import logging
import os
from tests.ops.op_test import OpTest, OpTestTool

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="pass_test")


class PassTest(OpTest):
    def __init__(self, *args, **kwargs):
        super(PassTest, self).__init__(*args, **kwargs)
        self.init_input_data()

    def init_input_data(self):
        self.feed_data = list()

    def build_program(self, builder, target):
        raise Exception("Not implemented.")

    def run_program(self):
        net_builder = NetBuilder("pass_test_netbuilder")

        inputs, outputs = self.build_program(net_builder, self.target)
        self.assertEqual(
            len(inputs), len(self.feed_data),
            "The feed data size not equal to program input size!")
        self.assertIsNotNone(outputs, "The program's output should not empty!")
        self.assertIsInstance(
            outputs[0], Variable,
            "The program's output should be list(cinn.frontend.Variable)")

        pass_prog = net_builder.build()
        return pass_prog, inputs, outputs

    def get_pass_outputs(self, passes):
        pass_prog, inputs, outputs = self.run_program()
        return self.get_cinn_output(pass_prog, self.target, inputs,
                                    self.feed_data, outputs, passes)

    def get_pass_size(self, passes):
        pass_prog, _, outputs = self.run_program()
        fetch_ids = {str(out) for out in outputs}
        logger.debug("Before pass {}:\n{}".format(passes, str(pass_prog)))
        op_num = pass_prog.apply_pass(fetch_ids, self.target, passes)
        logger.debug("After pass {}:\n{}".format(passes, str(pass_prog)))
        return op_num

    def check_pass_outputs(self,
                           pass_diff,
                           test_passes,
                           base_passes=[
                               "AutoCast", "Decomposer", "OpFusionPass",
                               "FusionMergePass"
                           ],
                           max_relative_error=1e-5,
                           all_equal=False,
                           equal_nan=False):
        base_pass_size = self.get_pass_size(base_passes)
        logger.debug(
            "Pass after base pass optimize has {} ops".format(base_pass_size))
        test_pass_size = self.get_pass_size(base_passes + test_passes)
        logger.debug(
            "Pass after base and test pass optimize has {} ops".format(
                test_pass_size))
        self.assertEqual(base_pass_size - test_pass_size, pass_diff,
                         "The pass not running as expected")

        cinn_no_pass_outputs = self.get_pass_outputs(base_passes)
        cinn_pass_outputs = self.get_pass_outputs(base_passes + test_passes)

        logger.debug("============ Check Outputs ============")
        self.check_results(cinn_no_pass_outputs, cinn_pass_outputs,
                           max_relative_error, all_equal, equal_nan)

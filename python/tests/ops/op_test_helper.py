# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

import itertools
import unittest


class TestCaseHelper():
    """
    构造测试用例输入参数的辅助类
    """

    def init_attrs(self):
        """
        初始化所有需要测试的属性
        """
        raise Exception("Not implemented.")

    def _flatten_tuple(self, cur_tuple):
        """
        将tuple中嵌套的字典展开
        """
        new_dict = []
        for cur_dict in cur_tuple:
            for k, v in cur_dict.items():
                new_dict.append((k, v))
        return dict(new_dict)

    def _init_cases(self):
        """
        生成所有的测试用例
        """
        self.all_cases = []
        attrs_cases = (dict(zip(self.attrs.keys(), values))
                       for values in itertools.product(*self.attrs.values()))
        for case in itertools.product(self.inputs, self.dtypes, attrs_cases):
            self.all_cases.append(self._flatten_tuple(case))

    def _make_all_classes(self):
        """
        动态构造所有测试类
        """
        self.init_attrs()
        self._init_cases()
        self.all_classes = []
        for i, case in enumerate(self.all_cases):
            self.all_classes.append(
                type(f'{self.class_name}{i}', (self.cls, ), {"case": case}))
        return self.all_classes

    def run(self):
        """
        运行所有的测试类
        """
        self._make_all_classes()
        test_suite = unittest.TestSuite()
        test_loader = unittest.TestLoader()
        for x in self.all_classes:
            test_suite.addTests(test_loader.loadTestsFromTestCase(x))
        runner = unittest.TextTestRunner()
        runner.run(test_suite)

#!/usr/bin/env python3

import sys

print("#include <string_view>\n")
print("namespace cinn::backends {")
print("std::string_view kRuntimeLlvmIr(")

print("R\"ROC(")
for i, line in enumerate(sys.stdin):
    print(line.strip())
print(")ROC\"")

print(");")
print("}  // namespace cinn::backends")

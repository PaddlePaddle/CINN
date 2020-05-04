#!/usr/bin/env python3

import sys
path = sys.argv[1]
out_path = sys.argv[2]

sys.stdout = open(out_path, 'w')

print("#pragma once\n")
print("#include <string_view>\n")
print("namespace cinn::backends {")
print("static std::string_view kRuntimeLlvmIr(")

print("R\"ROC(")

with open(path) as f:
    for i, line in enumerate(f):
        print(line.strip())
print(")ROC\"")

print(");")
print("}  // namespace cinn::backends")

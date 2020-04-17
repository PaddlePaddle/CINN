#!/usr/bin/env python3

import sys

print("#include <string_view>\n")
print("namespace cinn::backends {")
print("std::string_view kRuntimeLlvmIr(")

for i, line in enumerate(sys.stdin):
    r = line.replace('\\', '\\\\').replace("\"", "\\\"").rstrip('\n')
    print(f"  \"{r}\\n\"")

print(");")
print("}  // namespace cinn::backends")

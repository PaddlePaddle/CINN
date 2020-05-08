#!/usr/bin/env python3

import sys

def main():
    path = sys.argv[1]
    out_path = sys.argv[2]

    srcs = []
    srcs.append('#include <string_view>')
    #srcs.append('#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"\n')
    srcs.append('namespace cinn::backends {')
    srcs.append("static std::string_view kRuntimeLlvmIr(")
    srcs.append('R"ROC(')
    with open(path, 'r') as fr:
        srcs.append(fr.read())

    srcs.append(')ROC"')
    srcs.append(');')
    srcs.append('}  // namespace cinn::backends')
    with open(out_path, 'w') as fw:
        fw.write("\n".join(srcs))

if __name__ == "__main__":
    main()

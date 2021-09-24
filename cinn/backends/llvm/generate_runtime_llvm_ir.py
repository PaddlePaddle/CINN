#!/usr/bin/env python3

import sys
import subprocess


def main():
    path = sys.argv[1]
    out_path = sys.argv[2]

    srcs = []
    srcs.append('#include <absl/strings/string_view.h>')
    #srcs.append('#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"\n')
    srcs.append('namespace cinn::backends {')
    srcs.append("inline absl::string_view kRuntimeLlvmIr(")
    srcs.append('R"ROC(')
    with open(path, 'r') as fr:
        srcs.append(fr.read())

    srcs.append(')ROC"')
    srcs.append(');\n')

    cmd = "llvm-config --version"
    version = subprocess.check_output(
        cmd, shell=True).decode('utf-8').strip().split('.')
    srcs.append("struct llvm_version {")
    for v, n in zip(["major", "minor", "micro"], version):
        srcs.append(f"  static constexpr int k{v.title()} = {n};")
    srcs.append("};")

    srcs.append('}  // namespace cinn::backends')
    with open(out_path, 'w') as fw:
        fw.write("\n".join(srcs))


def get_clang_version():
    pass


if __name__ == "__main__":
    main()

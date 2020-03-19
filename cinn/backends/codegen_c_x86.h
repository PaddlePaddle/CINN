#pragma once

#include "cinn/backends/codegen_c.h"

namespace cinn {
namespace backends {

/**
 * C code generation with X86 instruction or math library support.
 */
class CodeGenCX86 : public CodeGenC {
 public:
  //! The X86 CPU supports some following features. We use SSE or AVX to accelerate the basic operations if forloop is
  //! vectorized.
  enum class Feature : int {
    None   = 0,
    SSE    = 1,       //! support SSE instruction set.
    AVX256 = 1 << 1,  // ! support AVX256 instruction set.
    AVX512 = 1 << 2,  // ! support AVX512 instruction set.
    BLAS   = 1 << 3,  // ! support BLAS library.
  };

  Feature feature{Feature::None};

  /**
   * constructor.
   * @param target The device.
   * @param features Features it supported.
   */
  CodeGenCX86(Target target, Feature feature) : CodeGenC(target), feature(feature) {}

 protected:
  void Visit(const ir::Add *op) override;
  void Visit(const ir::Sub *op) override;
  void Visit(const ir::Mul *op) override;
  void Visit(const ir::Div *op) override;
  void Visit(const ir::Mod *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::EQ *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::NE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::LT *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::LE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::GT *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::GE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::And *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::Or *op) override { CodeGenC::Visit(op); }

  void Visit(const ir::Load *op) override;
  void Visit(const ir::Store *op) override;

  //! Check the features.
  // @{
  bool SupportsSSE() { return static_cast<int>(feature) & static_cast<int>(Feature::SSE); }
  bool SupportsAVX256() { return static_cast<int>(feature) & static_cast<int>(Feature::AVX256); }
  bool SupportsAVX512() { return static_cast<int>(feature) & static_cast<int>(Feature::AVX512); }
  bool SupportsBLAS() { return static_cast<int>(feature) & static_cast<int>(Feature::BLAS); }
  // @}

  //! Print (and prepare) a argument in vectorize type, for example:
  // 3. -> set1(3.)
  // a[i:j] -> load_ps(a+i)
  void PrintVecInputArgument(const Expr *op);
  //! The output argument, such as the destination for Load.
  void PrintVecOutputArgument(const Expr *op);
  void PrintAbsAddr(const ir::Load *op);
  template <typename Op>
  void VisitBinaryOp(const Op *op, Expr a, Expr b, const std::string &op_repr);
};

template <typename Op>
void CodeGenCX86::VisitBinaryOp(const Op *op, Expr a, Expr b, const std::string &op_repr) {
  CHECK_EQ(a.type(), b.type());

  // scalar.
  if (a.type().lanes() == 1) {
    CodeGenC::Visit(op);
    return;
  }

  // TODO(Superjomn) Consider support BLAS.
  int bits = a.type().bits() * a.type().lanes();
  if (SupportsAVX512()) {
    CHECK_EQ(bits, 512) << "the bits of computation should be times of 512";
    os() << "cinn_avx512_" << op_repr << "(";
    PrintVecInputArgument(&a);
    os() << ", ";
    PrintVecInputArgument(&b);
    os() << ")";
  } else if (SupportsAVX256()) {
    CHECK_EQ(bits, 256) << "the bits of computation should be times of 256";
    os() << "cinn_avx256_" << op_repr << "(";
    PrintVecInputArgument(&a);
    os() << ", ";
    PrintVecInputArgument(&b);
    os() << ")";
  } else {
    NOT_IMPLEMENTED
  }
}

}  // namespace backends
}  // namespace cinn

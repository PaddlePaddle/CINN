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
  CodeGenCX86(Target target, Feature &feature) : CodeGenC(target), feature(feature) {}

 protected:
  void Visit(const ir::Add *op) override;
  void Visit(const ir::Sub *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::Mul *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::Div *op) override { CodeGenC::Visit(op); }
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
};

}  // namespace backends
}  // namespace cinn

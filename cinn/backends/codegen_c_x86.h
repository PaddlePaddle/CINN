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
  enum class Feature {
    SSE,     //! support SSE instruction set.
    AVX256,  // ! support AVX256 instruction set.
    AVX512,  // ! support AVX512 instruction set.
    BLAS,    // ! support BLAS library.
  };
  using features_t = std::vector<Feature>;

  /**
   * constructor.
   * @param target The device.
   * @param features Features it supported.
   */
  CodeGenCX86(Target target, const features_t &features) : CodeGenC(target) {}

 protected:
  void Visit(const ir::Add *op) override { CodeGenC::Visit(op); }
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
};

}  // namespace backends
}  // namespace cinn

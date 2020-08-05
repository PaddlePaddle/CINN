#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/hlir/instruction/compiler.h"
#include "cinn/hlir/instruction/computation.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/module.h"
#include "cinn/hlir/instruction/optimizer.h"

namespace cinn {
namespace python {

namespace hlir_instr = hlir::instruction;

struct py_instruction {
  py_instruction(hlir::instruction::Instruction* data) : data(data) {}  // NOLINT

  hlir::instruction::Instruction* data{};
};

struct py_computation {
  explicit py_computation(hlir::instruction::Computation* data) : data(data) {}

  hlir::instruction::Computation* data{};
};

struct py_shape {
  void add_int_dim(int x) { data_.AddDim(x); }

  void add_var_dim(const std::string& name) {
    ir::Var x(name, Int(32));
    data_.AddDim(x);
  }

  hlir::instruction::Shape get_raw() const { return data_; }

 private:
  hlir::instruction::Shape data_;
};

struct py_context {
  hlir_instr::Context data{};
};

struct py_computation_builder {
  explicit py_computation_builder(py_context& context, const std::string& name) : builder_(&context.data, name) {}

  py_instruction add_parameter(int param_offset, py_shape shape, const std::string& name, const std::string& dtype);

  py_instruction add_binary(const std::string& opr, py_instruction a, py_instruction b);

  py_instruction add_dot(py_instruction a, py_instruction b);

  std::unique_ptr<hlir_instr::Computation> build() { return builder_.Build(); }

 private:
  hlir::instruction::Computation::Builder builder_;
};

/**
 * Usage:
 *
 * py_module module;
 * module.create
 * module.create_sigmoid()
 */
struct py_module {
  explicit py_module(const std::string& name) : data(name) {}

  void add_computation(py_computation_builder& builder);  // NOLINT

  void add_entry_computation(py_computation_builder& builder);  // NOLINT

  hlir::instruction::Module data;
};

struct py_buffer {
  py_buffer(const std::vector<int>& shape, const std::string& dtype, const std::string& device, int data_align);

  static std::shared_ptr<py_buffer> from_numpy(pybind11::array array);
  pybind11::array numpy();

  ~py_buffer() {
    if (data_) {
      cinn_buffer_free(nullptr, data_);
      data_ = nullptr;
    }
  }

  cinn_buffer_t* raw() { return data_; }

 private:
  cinn_buffer_t* data_{};
};

struct py_args {
  void add_buffer(const std::shared_ptr<py_buffer>& buf) { data_.emplace_back(buf->raw()); }

  void add_int32(int32_t v) { data_.emplace_back(v); }

  int size() const { return data_.size(); }

  cinn_pod_value_t* raw() { return data_.data(); }

 private:
  std::vector<cinn_pod_value_t> data_;
};

struct py_compiler {
  py_compiler();

  void compile(py_module& module) {
    hlir_instr::Optimizer().Run(&module.data);
    data_->Compile(&module.data);
  }

  void eval(const std::string& fn_name, py_args& args) { data_->Eval(fn_name, args.raw(), args.size()); }
  void eval_main(py_module& module, py_args& args) { data_->Eval(&module.data, args.raw(), args.size()); }

 private:
  std::unique_ptr<hlir_instr::Compiler> data_;
};

}  // namespace python
}  // namespace cinn

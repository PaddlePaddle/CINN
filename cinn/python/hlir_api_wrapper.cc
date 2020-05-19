#include "cinn/python/hlir_api_wrapper.h"

namespace cinn {
namespace python {

py_instruction python::py_computation_builder::add_binary(const std::string& opr,
                                                          python::py_instruction a,
                                                          python::py_instruction b) {
  auto shape = a.data->shape();
  hlir_instr::Instruction* ptr{};
  if (opr == "add") {
    ptr = builder_.AddInstruction(
        hlir_instr::Instruction::CreateBinary(hlir_instr::InstrCode::Add, a.data, b.data, shape));
  } else if (opr == "sub") {
    ptr = builder_.AddInstruction(
        hlir_instr::Instruction::CreateBinary(hlir_instr::InstrCode::Sub, a.data, b.data, shape));
  } else if (opr == "mul") {
    ptr = builder_.AddInstruction(
        hlir_instr::Instruction::CreateBinary(hlir_instr::InstrCode::Mul, a.data, b.data, shape));
  } else if (opr == "div") {
    ptr = builder_.AddInstruction(
        hlir_instr::Instruction::CreateBinary(hlir_instr::InstrCode::Div, a.data, b.data, shape));
  } else {
    NOT_IMPLEMENTED;
  }

  return py_instruction(ptr);
}

python::py_instruction python::py_computation_builder::add_parameter(int param_offset,
                                                                     python::py_shape shape,
                                                                     const std::string& name,
                                                                     const std::string& dtype) {
  // parameter config
  hlir::instruction::ParameterConfig config;
  if (dtype == "float32") {
    config.type = Float(32);
  } else if (dtype == "float64") {
    config.type = Float(64);
  } else {
    NOT_IMPLEMENTED
  }

  auto ptr =
      builder_.AddInstruction(hlir_instr::Instruction::CreateParameter(param_offset, shape.get_raw(), name, config));
  return py_instruction(ptr);
}

py_instruction py_computation_builder::add_dot(py_instruction a, py_instruction b) {
  auto* ptr = builder_.AddInstruction(hlir_instr::Instruction::CreateDot(a.data, b.data));
  return py_instruction(ptr);
}

void py_module::add_entry_computation(py_computation_builder& builder) { data.AddEntryComputation(builder.build()); }

void py_module::add_computation(py_computation_builder& builder) { data.AddComputation(builder.build()); }

std::string NumpyDtypeToCinn(pybind11::dtype type) {
  std::string t;
  if (type == pybind11::dtype::of<float>()) {
    t = "float32";
  } else if (type == pybind11::dtype::of<double>()) {
    t = "float64";
  } else if (type == pybind11::dtype::of<int32_t>()) {
    t = "int32";
  } else if (type == pybind11::dtype::of<int64_t>()) {
    t = "int64";
  } else {
    NOT_IMPLEMENTED
  }
  return t;
}

std::shared_ptr<py_buffer> py_buffer::from_numpy(pybind11::array array) {
  std::vector<int> shape;
  for (int i = 0; i < array.ndim(); i++) {
    LOG(INFO) << "shape " << i << " " << array.shape(i);
    shape.push_back(array.shape(i));
  }

  std::string type = NumpyDtypeToCinn(array.dtype());

  auto buffer = std::make_shared<py_buffer>(shape, type, "x86", 32);
  cinn_buffer_malloc(nullptr, buffer->data_);
  CHECK(buffer->data_->host_memory);

  std::memcpy(static_cast<void*>(buffer->data_->host_memory), array.mutable_data(), buffer->data_->memory_size);
  return buffer;
}

pybind11::array py_buffer::to_numpy() {
  pybind11::dtype t;
  if (data_->type == cinn_float32_t()) {
    t = pybind11::dtype::of<float>();
  } else if (data_->type == cinn_float64_t()) {
    t = pybind11::dtype::of<double>();
  } else if (data_->type == cinn_int32_t()) {
    t = pybind11::dtype::of<int32_t>();
  } else if (data_->type == cinn_int64_t()) {
    t = pybind11::dtype::of<int64_t>();
  } else {
    NOT_IMPLEMENTED
  }

  pybind11::array::ShapeContainer shape(data_->dims, data_->dims + data_->dimensions);
  pybind11::array py_data(t, std::move(shape));
  CHECK(data_->host_memory);
  std::memcpy(py_data.mutable_data(), data_->host_memory, data_->memory_size);
  return py_data;
}

py_buffer::py_buffer(const std::vector<int>& shape,
                     const std::string& dtype,
                     const std::string& device,
                     int data_align) {
  cinn_device_kind_t d;
  if (device == "host") {
    d = cinn_x86_device;
  } else if (device == "x86") {
    d = cinn_x86_device;
  } else {
    NOT_IMPLEMENTED
  }

  cinn_type_t t;
  if (dtype == "float32") {
    t = cinn_float32_t();
  } else if (dtype == "float64") {
    t = cinn_float64_t();
  } else if (dtype == "int32") {
    t = cinn_int32_t();
  } else if (dtype == "int64") {
    t = cinn_int64_t();
  } else {
    NOT_IMPLEMENTED
  }

  data_ = cinn_buffer_t::new_(d, t, shape, data_align);
}

}  // namespace python
}  // namespace cinn

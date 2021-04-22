#include <memory>
#include <string>
#include <vector>

#include "cinnrt/tensor/dense_host_tensor.h"

namespace cinnrt {

class ConfigBase {
  std::string model_dir_;
  std::string mlir_path_;
  std::vector<std::string> shared_libs_;

 public:
  ConfigBase() = default;
  void set_model_dir(const std::string& model_dir) { model_dir_ = model_dir; };
  const std::string& model_dir() const { return model_dir_; }

  void set_mlir_path(const std::string& mlir_path) { mlir_path_ = mlir_path; };
  const std::string& mlir_path() const { return mlir_path_; }

  void set_shared_libs(const std::vector<std::string>& shared_libs) { shared_libs_ = shared_libs; };
  const std::vector<std::string>& shared_libs() const { return shared_libs_; }

  virtual ~ConfigBase() = default;
};

class CinnRtConfig : public ConfigBase {};

class CinnRtPredictor {
 public:
  CinnRtPredictor();
  ~CinnRtPredictor();
  void Run();
  int Init(const CinnRtConfig& config);
  int GetInputNum();
  tensor::DenseHostTensor* GetInput(int i);
  int GetOutputNum();
  tensor::DenseHostTensor* GetOutput(int i);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<CinnRtPredictor> CreateCinnRtPredictor(const CinnRtConfig& config);

}  // namespace cinnrt

#pragma once
#include <glog/logging.h>
#include <string>
#include <vector>

namespace cinn {
namespace frontend {
namespace paddle {

class BlockDescReadAPI {
 public:
  virtual int32_t Idx() const             = 0;
  virtual int32_t ParentIdx() const       = 0;
  virtual size_t VarsSize() const         = 0;
  virtual size_t OpsSize() const          = 0;
  virtual int32_t ForwardBlockIdx() const = 0;

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T const* GetVar(int32_t idx) const;

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T const* GetOp(int32_t idx) const;

  virtual ~BlockDescReadAPI() = default;
};

class BlockDescWriteAPI {
 public:
  virtual void SetIdx(int32_t idx) { NotImplemented(); }
  virtual void SetParentIdx(int32_t idx) { NotImplemented(); }
  virtual void ClearVars() { NotImplemented(); }
  virtual void ClearOps() { NotImplemented(); }
  virtual void SetForwardBlockIdx(int32_t idx) { NotImplemented(); }

  template <typename T>
  T* AddVar() {
    NotImplemented();
    return nullptr;
  }

  template <typename T>
  T* AddOp() {
    NotImplemented();
    return nullptr;
  }

  virtual ~BlockDescWriteAPI() = default;

 private:
  void NotImplemented() const { LOG(FATAL) << "BlockDescWriteAPI is not available in model read-only mode."; }
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class BlockDescAPI : public BlockDescReadAPI, public BlockDescWriteAPI {
 public:
  virtual ~BlockDescAPI() = default;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace cinn

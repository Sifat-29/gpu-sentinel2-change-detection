// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Error checking and RAII wrappers for CUDA resources. Every CUDA runtime
// call in the GPU backend goes through CD_CUDA_CHECK, which turns an error
// code into a cd::Status that names the failing call, file and line.
#ifndef SRC_CUDA_CUDA_CHECK_CUH_
#define SRC_CUDA_CUDA_CHECK_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>

#include "status.h"

namespace cd {

// Converts a CUDA error code into a Status. `expr` is the text of the failing
// call, used in the message together with the source location.
inline Status CudaStatus(cudaError_t error, const char* expr, const char* file,
                         int line) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::Error(std::string(file) + ":" + std::to_string(line) + ": " +
                       expr + " failed: " + cudaGetErrorName(error) + " (" +
                       cudaGetErrorString(error) + ")");
}

}  // namespace cd

// Evaluates a CUDA runtime call and returns its error as a cd::Status from
// the enclosing function (which must return cd::Status).
#define CD_CUDA_CHECK(expr)                                              \
  do {                                                                   \
    const cudaError_t cd_cuda_error = (expr);                            \
    if (cd_cuda_error != cudaSuccess) {                                  \
      return ::cd::CudaStatus(cd_cuda_error, #expr, __FILE__, __LINE__); \
    }                                                                    \
  } while (0)

// Returns early from a Status-returning function when `expr` is an error.
#define CD_RETURN_IF_ERROR(expr)           \
  do {                                     \
    const ::cd::Status cd_status = (expr); \
    if (!cd_status.ok()) {                 \
      return cd_status;                    \
    }                                      \
  } while (0)

namespace cd {

// Owning wrapper of a cudaMalloc allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Free(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { Swap(&other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    Free();
    Swap(&other);
    return *this;
  }

  // Replaces the allocation. Returns the raw CUDA error so that callers can
  // tell an out-of-memory failure (cudaErrorMemoryAllocation) from others.
  cudaError_t Allocate(size_t count) {
    Free();
    if (count == 0) {
      return cudaSuccess;
    }
    void* pointer = nullptr;
    const cudaError_t error = cudaMalloc(&pointer, count * sizeof(T));
    if (error != cudaSuccess) {
      cudaGetLastError();  // allocation errors are not sticky; clear them
      return error;
    }
    ptr_ = static_cast<T*>(pointer);
    count_ = count;
    return cudaSuccess;
  }

  void Free() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
    ptr_ = nullptr;
    count_ = 0;
  }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  void Swap(DeviceBuffer* other) {
    std::swap(ptr_, other->ptr_);
    std::swap(count_, other->count_);
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

// Owning wrapper of host memory: page-locked (cudaMallocHost) when possible,
// ordinary pageable memory otherwise. Pinned memory is required for transfers
// that overlap with kernels; pageable memory still works, only slower.
template <typename T>
class HostBuffer {
 public:
  HostBuffer() = default;
  ~HostBuffer() { Free(); }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  HostBuffer(HostBuffer&& other) noexcept { Swap(&other); }
  HostBuffer& operator=(HostBuffer&& other) noexcept {
    Free();
    Swap(&other);
    return *this;
  }

  // Replaces the allocation; returns false only when even pageable memory is
  // not available. pinned() tells which kind was obtained.
  bool Allocate(size_t count) {
    Free();
    if (count == 0) {
      return true;
    }
    void* pointer = nullptr;
    if (cudaMallocHost(&pointer, count * sizeof(T)) == cudaSuccess) {
      pinned_ = true;
    } else {
      cudaGetLastError();  // clear the failed pinned allocation
      pointer = std::malloc(count * sizeof(T));
      if (pointer == nullptr) {
        return false;
      }
      pinned_ = false;
    }
    ptr_ = static_cast<T*>(pointer);
    count_ = count;
    return true;
  }

  void Free() {
    if (ptr_ != nullptr) {
      if (pinned_) {
        cudaFreeHost(ptr_);
      } else {
        std::free(ptr_);
      }
    }
    ptr_ = nullptr;
    count_ = 0;
    pinned_ = false;
  }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }
  bool pinned() const { return pinned_; }

 private:
  void Swap(HostBuffer* other) {
    std::swap(ptr_, other->ptr_);
    std::swap(count_, other->count_);
    std::swap(pinned_, other->pinned_);
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
  bool pinned_ = false;
};

// Owning wrapper of a non-blocking CUDA stream (does not synchronise with the
// legacy default stream).
class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { Destroy(); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  Status Create() {
    Destroy();
    CD_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    return Status::Ok();
  }

  void Destroy() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
  }

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// Owning wrapper of a CUDA event with timing enabled.
class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() { Destroy(); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  Status Create() {
    Destroy();
    CD_CUDA_CHECK(cudaEventCreate(&event_));
    return Status::Ok();
  }

  void Destroy() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
    event_ = nullptr;
  }

  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace cd

#endif  // SRC_CUDA_CUDA_CHECK_CUH_

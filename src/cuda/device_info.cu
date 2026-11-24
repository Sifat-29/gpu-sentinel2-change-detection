// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Device queries for the GPU backend. Compiled only in GPU builds; the
// CPU-only build gets the same two functions from gpu_stub.cc.
#include <cuda_runtime.h>

#include <string>

#include "cuda/gpu_backend.h"

namespace cd {
namespace {

Status CudaError(const char* what, cudaError_t err) {
  return Status::Error(std::string(what) + ": " + cudaGetErrorString(err));
}

}  // namespace

bool GpuAvailable() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    // Clear the sticky error so later CUDA calls are not affected.
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

Status GpuDeviceInfo(int device, GpuInfo* info) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return CudaError("cudaGetDeviceCount", err);
  }
  if (device < 0 || device >= count) {
    return Status::Error("GPU index " + std::to_string(device) +
                         " is out of range (" + std::to_string(count) +
                         " device(s) found)");
  }
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return CudaError("cudaGetDeviceProperties", err);
  }
  err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return CudaError("cudaSetDevice", err);
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
    return CudaError("cudaMemGetInfo", err);
  }
  int driver_version = 0;
  int runtime_version = 0;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  info->name = prop.name;
  info->cc_major = prop.major;
  info->cc_minor = prop.minor;
  info->sm_count = prop.multiProcessorCount;
  info->total_mem = total_bytes;
  info->free_mem = free_bytes;
  info->driver_version = driver_version;
  info->runtime_version = runtime_version;
  return Status::Ok();
}

}  // namespace cd

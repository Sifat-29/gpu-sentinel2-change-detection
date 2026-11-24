// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Portability macro for functions shared by host (CPU) and device (GPU) code.
#ifndef SRC_HOST_DEVICE_H_
#define SRC_HOST_DEVICE_H_

// CD_HOST_DEVICE marks a function that is compiled for both the CPU and the
// GPU when nvcc is the compiler, and is empty for plain C++ compilers. All
// per-pixel maths uses it so the same source runs on every backend.
#ifdef __CUDACC__
#define CD_HOST_DEVICE __host__ __device__
#else
#define CD_HOST_DEVICE
#endif

#endif  // SRC_HOST_DEVICE_H_

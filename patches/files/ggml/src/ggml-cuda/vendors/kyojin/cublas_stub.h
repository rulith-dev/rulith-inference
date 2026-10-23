#pragma once
// strixllama: the CUDA runtime and cuBLAS names the vendored exllamav3 headers use, which torch's hipify maps in the
// project they come from. Only mmh-moe.cu includes these headers; nothing here calls a BLAS library.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaStream_t hipStream_t
#define cudaPeekAtLastError hipPeekAtLastError
typedef int cublasStatus_t;
enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE,
       CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR,
       CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR };
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

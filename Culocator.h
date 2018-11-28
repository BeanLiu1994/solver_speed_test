#pragma once

#include <cuda_runtime.h>
cudaError_t _cu_malloc(void** device_ptr, size_t size_byte);
cudaError_t _cu_free(void* device_ptr);
cudaError_t _cu_syncDevice();
cudaError_t _cu_memset(void* device_ptr, size_t size_byte, int val);
cudaError_t _cu_getResult(void* device_ptr, size_t size_byte, void* OutPtr);
cudaError_t _cu_getResult(void* device_ptr, size_t size_byte, const void* OutPtr);
cudaError_t _cu_copyToDevice(void* device_ptr, size_t size_byte, const void* InPtr);


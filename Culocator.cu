
#include <cuda_runtime.h>
#include <stdexcept>
#include "CudaManager.h"


cudaError_t _cu_malloc(void** device_ptr, size_t size_byte)
{
	return gpuErrchk(cudaMalloc(device_ptr, size_byte));
}
cudaError_t _cu_free(void* device_ptr)
{
	return gpuErrchk(cudaFree(device_ptr));
}
cudaError_t _cu_syncDevice()
{
	return gpuErrchk(cudaDeviceSynchronize());
}
cudaError_t _cu_memset(void* device_ptr, size_t size_byte, int val)
{
	return gpuErrchk(cudaMemset(device_ptr, val, size_byte));
}
cudaError_t _cu_getResult(void* device_ptr, size_t size_byte, void* OutPtr)
{
	if (OutPtr == nullptr)
		throw std::runtime_error("copy with nullptr");
	return gpuErrchk(cudaMemcpy(OutPtr, device_ptr, size_byte, cudaMemcpyDeviceToHost));
}

cudaError_t _cu_copyToDevice(void* device_ptr, size_t size_byte, const void* InPtr)
{
	if (InPtr == nullptr)
		throw std::runtime_error("copy with nullptr");
	return gpuErrchk(cudaMemcpy(device_ptr, InPtr, size_byte, cudaMemcpyHostToDevice));
}
cudaError_t _cu_getResult(void* device_ptr, size_t size_byte, const void* OutPtr)
{
	// do nothing
	return cudaSuccess;
}
#include "CudaManager.h"
#include <iostream>
#include <string>

int cudaInitializer::dev = -1;
cudaInitializer::cudaInitializer()
{

}
int cudaInitializer::Init()
{
	if (!cudaInitializer::CudaOK())
	{
		try
		{
			dev = findCudaDevice(0, nullptr);
		}
		catch (std::exception& e)
		{
			std::cerr << std::string(e.what()) << std::endl;
			return dev;
		}
		cudaDeviceProp deviceProp;
		gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));
		//printf("[Cuda Initial Succeed] GPU Device %d: \"%s\" with compute capability %d.%d\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	return dev;
}

bool cudaInitializer::CudaOK()
{
	return dev != -1;
}



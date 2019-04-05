CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(FaceReconstruction)

add_compile_options(-g -O3)

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 5.2 6.0)  
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED) # mkl
find_package(LAPACK REQUIRED) # mkl
find_package(SuiteSparse)
find_package(SuiteSparse REQUIRED)
find_package(Eigen3 REQUIRED)

if(BLAS_mkl_intel_lp64_dll_LIBRARY)
	find_path(MKL_INCLUDE NAMES "mkl.h" PATHS ${BLAS_mkl_intel_lp64_dll_LIBRARY}/../../../ PATH_SUFFIXES include)
	include_directories(${MKL_INCLUDE})
endif()
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE_DIR})

list(APPEND CUDA_NVCC_FLAGS "-rdc=true" "-std=c++11")
file(GLOB PATH_SRCS_SpeedTest *.cu *.cpp)
CUDA_ADD_EXECUTABLE(SpeedTest ${PATH_SRCS_SpeedTest})
set_property(TARGET SpeedTest PROPERTY CXX_STANDARD 11)
target_link_libraries(SpeedTest ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY} ${SUITESPARSE_LIBRARIES} ${UMFPACK_LIBRARY} ${BLAS_LIBRARIES})

message("${BLAS_LIBRARIES}")
if(NOT WIN32)
	target_link_libraries(SpeedTest z)
endif()



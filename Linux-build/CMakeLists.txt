CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(FaceReconstruction)

set(CMAKE_BUILD_TYPE Release)
# 宏1为config时表示需要读取cmake配置的宏
# 宏2为libcurl实际使用了静态库的宏，配合curl头文件使用
add_compile_options(-g -O3)

include(FindDeps.cmake)

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})


set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DCMAKE_GENERATE" "-rdc=true" "-std=c++11")
file(GLOB PATH_SRCS_SpeedTest *.cu *.cpp)
CUDA_ADD_EXECUTABLE(SpeedTest ${PATH_SRCS_SpeedTest})

set_property(TARGET SpeedTest PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET SpeedTest PROPERTY CXX_STANDARD 11)

target_link_libraries(SpeedTest ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY} Eigen3::Eigen ${SuiteSparse_LIBRARIES} ${MKL_LIBRARIES} z)



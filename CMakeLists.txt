cmake_minimum_required(VERSION 3.0)
project("Diffusion Simulation")

find_package(CUDA REQUIRED)

file(GLOB SRC "src/*.cpp" "src/*.cu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(CUDA_FOUND)
  message("CUDA Version:" ${CUDA_VERSION})

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -Xcompiler -std=c++11")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -lineinfo --ptxas-options=-v")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -G -g")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -O3")
  endif()

  CUDA_ADD_EXECUTABLE(main ${SRC})
endif()

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SA)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(CUDA_FOUND)
  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  file( GLOB  cu  *.cu)
  file( GLOB  hdr *.hpp *.h )
  SET (CPP_FILES sa_ext.cpp)
  CUDA_ADD_EXECUTABLE(SA ${CPP_FILES} ${cu} ${hdr})
  target_link_libraries(SA "${TORCH_LIBRARIES}")
  set_property(TARGET SA PROPERTY CXX_STANDARD 11)
else(CUDA_FOUND)
  message("CUDA is not found!")
endif()

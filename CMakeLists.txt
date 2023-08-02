cmake_minimum_required(VERSION 3.17)

project(onnx_cpp_container CXX CUDA)

# Set the C++ standard (optional, change to your desired version)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
endif()

# onnxruntime providers
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)

set(ONNXRUNTIME_INCLUDDIRS ${ONNXRUNTIME_ROOTDIR}/include                           # Pre-built package
                           ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime               # Linux local install to /usr/local
                           ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session) # Windows local install

set(ONNXRUNTIME_LIBS onnxruntime onnxruntime_providers_cuda onnxruntime_providers_shared onnxruntime_providers_tensorrt)

link_directories("${ONNXRUNTIME_ROOTDIR}/lib" ${OpenCV_INCLUDE_DIRS})

# Find the ONNX Runtime package using vcpkg
# find_package(onnxruntime-gpu CONFIG REQUIRED)
include(FetchContent)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(CUDA REQUIRED)

set(LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/onnx_inference.cpp)
set(LIB_HEADERS ${CMAKE_CURRENT_LIST_DIR}/src/onnx_inference.h)

add_library(lib_onnx_example ${LIB_SOURCES} ${LIB_HEADERS})
add_executable(onnx_example ./src/main.cpp)

target_include_directories(onnx_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})
target_include_directories(lib_onnx_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(lib_onnx_example PUBLIC CUDA cudart ${ONNXRUNTIME_LIBS})
target_link_libraries(onnx_example PRIVATE lib_onnx_example gflags ${OpenCV_LIBS} CUDA cudart ${ONNXRUNTIME_LIBS})

add_definitions(-DUSE_CUDA -DUSE_TENSORRT)
target_compile_definitions(onnx_example PRIVATE -DGPU)
target_compile_definitions(lib_onnx_example PUBLIC -DGPU)
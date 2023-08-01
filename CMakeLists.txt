cmake_minimum_required(VERSION 3.17)

project(onnx_cpp_container C CXX CUDA)

# Set the C++ standard (optional, change to your desired version)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

# onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" ON)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" ON)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

if(onnxruntime_USE_CUDA OR onnxruntime_USE_TENSORRT)
  include(CheckLanguage)
  check_language(CUDA)
 endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib" ${OpenCV_INCLUDE_DIRS})

# Find the ONNX Runtime package using vcpkg
# find_package(onnxruntime-gpu CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/src/*.c ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*/*.cpp)
list(REMOVE_ITEM sources
  ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h ${CMAKE_CURRENT_LIST_DIR}/src/*/*.h)

add_library(lib_onnx_example ${sources} ${headers})
add_executable(onnx_example ./src/main.cpp)

target_include_directories(onnx_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})
target_include_directories(lib_onnx_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_example PRIVATE lib_onnx_example)
target_link_libraries(lib_onnx_example PUBLIC onnxruntime CUDA cudart)
target_link_libraries(onnx_example PRIVATE onnxruntime gflags ${OpenCV_LIBS} CUDA cudart)

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
  target_compile_definitions(onnx_example PRIVATE -DGPU)
  target_compile_definitions(lib_onnx_example PUBLIC -DGPU)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
  target_compile_definitions(onnx_example PRIVATE -DGPU)
  target_compile_definitions(lib_onnx_example PUBLIC -DGPU)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()
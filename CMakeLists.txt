cmake_minimum_required(VERSION 3.12)
project(OnnxExample)

# Set the C++ standard (optional, change to your desired version)
set(CMAKE_CXX_STANDARD 17)

# onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" ON)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" ON)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

# Find the ONNX Runtime package using vcpkg
# find_package(onnxruntime-gpu CONFIG REQUIRED)
find_package(opencv REQUIRED)
find_package(onnxruntime-gpu REQUIRED)
	
add_executable(onnx_example ./src/onnx_inference.cpp)

# Link against the ONNX Runtime library
target_link_libraries(onnx_example PRIVATE onnxruntime)
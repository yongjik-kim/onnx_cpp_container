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

# Manually find onnxruntime
# TODO: Make find_package work with vcpkg

set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOTDIR}/include                           # Pre-built package
                             ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime               # Linux local install to /usr/local
                             ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session) # Windows local install

# Find the ONNX Runtime package using vcpkg
# find_package(onnxruntime-gpu CONFIG REQUIRED)
include(FetchContent)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(CUDA REQUIRED)

add_library(lib_onnx ./src/onnx_inference.cpp ./src/onnx_inference.h)
add_executable(resnet_example ./src/resnet_ex.cpp)
add_executable(pp_yolo_e_example ./src/pp_yolo_e_ex.cpp)

target_include_directories(resnet_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_include_directories(pp_yolo_e_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_include_directories(lib_onnx PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

target_link_directories(lib_onnx PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib")
target_link_libraries(lib_onnx PUBLIC CUDA cudart onnxruntime)
target_link_libraries(resnet_example PRIVATE lib_onnx gflags ${OpenCV_LIBS} CUDA cudart onnxruntime)
target_link_libraries(pp_yolo_e_example PRIVATE lib_onnx gflags ${OpenCV_LIBS} CUDA cudart onnxruntime)

add_definitions(-DUSE_CUDA -DUSE_TENSORRT)

# manually copy the .dlls because idk...

set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll
				     ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_cuda.dll
				     ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_shared.dll
				     ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_tensorrt.dll)

function(copy_ort_dlls target_name)
  if (MSVC)
      foreach(ORT_DLL ${ONNXRUNTIME_LIBS})
          add_custom_command(TARGET ${target_name} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
      endforeach()
  endif()
endfunction()

copy_ort_dlls(resnet_example)
copy_ort_dlls(pp_yolo_e_example)
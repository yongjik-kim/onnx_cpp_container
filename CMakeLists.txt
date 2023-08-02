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
                             ${ONNXRUNTIME_ROOTDIR}/include/lib
                             ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime               # Linux local install to /usr/local
                             ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session) # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib" ${OpenCV_INCLUDE_DIRS})


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

target_include_directories(onnx_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_include_directories(lib_onnx_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

target_link_libraries(lib_onnx_example PUBLIC CUDA cudart onnxruntime)
target_link_libraries(onnx_example PRIVATE lib_onnx_example gflags ${OpenCV_LIBS} CUDA cudart onnxruntime)

add_definitions(-DUSE_CUDA -DUSE_TENSORRT)
target_compile_definitions(onnx_example PRIVATE -DGPU)
target_compile_definitions(lib_onnx_example PUBLIC -DGPU)

copy_ort_dlls(onnx_example)
/*
Example of running onnx model using C++ api.

Copyright (C) 2023, Yongjik Kim
MIT License
*/

#include "onnx_inference.h"

#include <cuda_runtime.h>
// #include <gflags/gflags.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// DEFINE_string(provider, "CPU", "Execution provider: CPU, CUDA, or TensorRT");
// DEFINE_string(model_file, "yolo.onnx", "Directory to ONNX Model file");
// DEFINE_int(device, 0, "Device number to run inference on");

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

// Before using your model, it is highly recommended to use visualizer to check
// the model structure in detail! For example, use Netron:
// https://github.com/lutzroeder/netron

OnnxContainer::OnnxContainer(ORTCHAR_T* model_path,
    std::string execution_provider, std::vector<int64_t> input_shape,
    std::vector<int64_t> output_shape, float* input_arr, float* output_arr)
    : env_(nullptr),
      session_options_(),
      session_(nullptr),
      input_tensor_(nullptr),
      output_tensor_(nullptr),
      input_shape_({}),
      output_shape_({}),
      execution_provider_(ExecutionProviders::CPU),
      input_size_(1),
      output_size_(1),
      input_arr_(nullptr),
      output_arr_(nullptr),
      run_options_(nullptr),
      binding_(nullptr)
{
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);  // Change to ORT_ENABLE_BASIC if
                                                // gone wrong

  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");

  input_shape_ = input_shape;    // If image, in NCHW format
  output_shape_ = output_shape;  // in NCHW format
  input_size_ = vectorProduct(input_shape_);
  output_size_ = vectorProduct(output_shape_);
  input_arr_ = input_arr;
  output_arr_ = output_arr;
  model_path_ = model_path;

  SetUpEp(execution_provider);
};

OnnxContainer::~OnnxContainer()
{
  binding_.ClearBoundInputs();
  binding_.ClearBoundOutputs();
};

void OnnxContainer::SetUpEp(std::string execution_provider)
{
  if (execution_provider == "CPU" || execution_provider == "cpu" ||
      execution_provider == "Cpu")
  {
    execution_provider_ = ExecutionProviders::CPU;
    SetUpCpu();
  }
  else if (execution_provider == "CUDA" || execution_provider == "Cuda" ||
           execution_provider == "cuda")
  {
    execution_provider_ = ExecutionProviders::CUDA;
    SetUpCuda();
  }
  else if (execution_provider == "TensorRT" || execution_provider == "trt" ||
           execution_provider == "Tensorrt" ||
           execution_provider == "tensorrt" || execution_provider == "TRT")
  {
    execution_provider_ = ExecutionProviders::TENSORRT;
    SetUpTensorrt();
  }
  else
  {
    throw std::runtime_error("Unknown execution provider, exiting...");
  }
}

void OnnxContainer::SetUpCpu()
{
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_arr_,
      input_shape_.size(), input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, output_arr_,
      output_size_, output_shape_.data(), output_shape_.size());

  Ort::Session session(env_, model_path_, session_options_);
};

void OnnxContainer::SetUpCuda()
{
  auto const& api = Ort::GetApi();
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  std::unique_ptr<OrtCUDAProviderOptionsV2,
      decltype(api.ReleaseCUDAProviderOptions)>
      rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
      static_cast<OrtSessionOptions*>(session_options_),
      rel_cuda_options.get()));

  SetUpGpuIoBindings();
};

void OnnxContainer::SetUpTensorrt()
{
  auto const& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* trt_options;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
  std::unique_ptr<OrtTensorRTProviderOptionsV2,
      decltype(api.ReleaseTensorRTProviderOptions)>
      rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
      static_cast<OrtSessionOptions*>(session_options_),
      rel_trt_options.get()));

  SetUpGpuIoBindings();
};

void OnnxContainer::SetUpGpuIoBindings()
{
  // As binding_ is a class member, you have to instantiate another class if you
  // want to change the provider.
  session_ = Ort::Session(env_, model_path_, session_options_);
  Ort::MemoryInfo info_cuda(
      "cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session_, info_cuda);

  auto input_data = cuda_allocator.GetAllocation(input_size_ * sizeof(float));
  auto output_data = cuda_allocator.GetAllocation(output_size_ * sizeof(float));
  cudaMemcpy(input_data.get(), input_arr_, sizeof(float) * input_size_,
      cudaMemcpyHostToDevice);
  cudaMemcpy(output_data.get(), output_arr_, sizeof(float) * output_size_,
      cudaMemcpyHostToDevice);
  input_tensor_ = Ort::Value::CreateTensor(info_cuda, input_arr_, input_size_,
      input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor(info_cuda, output_arr_,
      output_size_, output_shape_.data(), input_shape_.size());

  binding_ = Ort::IoBinding(session_);
  binding_.BindInput("input", input_tensor_);
  binding_.BindOutput("output", output_tensor_);

  // One regular run for necessary memory allocation and graph capturing
  session_.Run(Ort::RunOptions(), binding_);
};

void OnnxContainer::Run() { session_.Run(run_options_, binding_); }

int main(int argc, char* argv[])
{
  /*
  // Initializing meta variables
  int64_t const input_c(3), input_h(224), input_w(224);
  int64_t const output_l(1000);
  std::vector<int64_t> input_shape{input_c, input_h, input_w};
  std::vector<int64_t> output_shape{output_l};
  float *input_arr, *output_arr;
  input_arr = new float[input_c * input_h * input_w];
  output_arr = new float[output_l];
  memset(input_arr, 0., input_c * input_h * input_w * sizeof(float));
  memset(output_arr, 0., output_l);


  // Parsing model, input file

  ORTCHAR_T* model_name = L"resnet50-v2-7.onnx";
  std::string image_path = "example.jpg";

  cv:Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  
  OnnxContainer engine(
      model_name, "CUDA", input_shape, output_shape, input_arr, output_arr);
  engine.Run();

  delete[] input_arr;
  delete[] output_arr;

  return 0;
  */
};
/*
Example of running onnx model using C++ api.

Copyright (C) 2023, Yongjik Kim
MIT License
*/

#include "onnx_inference.h"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

// Before using your model, it is highly recommended to use visualizer to check
// the model structure in detail! For example, use Netron:
// https://github.com/lutzroeder/netron

OnnxContainer::OnnxContainer(ORTCHAR_T* model_path, const char* input_node,
    const char* output_node, std::string execution_provider,
    std::vector<int64_t> input_shape, std::vector<int64_t> output_shape,
    float* input_arr, float* output_arr)
    : env_(ORT_LOGGING_LEVEL_WARNING, "test"),
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
      input_node_(input_node),
      output_node_(output_node),
      run_options_(),
      binding_(nullptr),
      output_device_data_(nullptr)
{
  session_options_ = Ort::SessionOptions();
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);  // Change to ORT_ENABLE_BASIC
                                                  // if gone wrong

  /*
  // Temporary setup to parse input / output
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXParsingApp");
  Ort::SessionOptions temp_options;
  Ort::Session session(env, model_path, temp_options);
  Ort::AllocatorWithDefaultOptions temp_allocator;  // Create an allocator
  const Ort::TypeInfo input_type_info =
      session.GetInputTypeInfo(0);  // Get the input type info
  const Ort::TypeInfo output_type_info =
      session.GetOutputTypeInfo(0);  // Get the output type info

  Ort::TensorTypeAndShapeInfo input_tensor_info =
      input_type_info.GetTensorTypeAndShapeInfo();  // Get tensor info for input
  Ort::TensorTypeAndShapeInfo output_tensor_info =
      output_type_info
          .GetTensorTypeAndShapeInfo();  // Get tensor info for output

  size_t num_input_dims =
      input_tensor_info
          .GetDimensionsCount();  // Get the number of dimensions for the input
  size_t num_output_dims =
      output_tensor_info
          .GetDimensionsCount();  // Get the number of dimensions for the output

  std::cout << "Input size: ";
  for (size_t i = 0; i < num_input_dims; ++i)
  {
    std::cout << input_tensor_info.GetShape()[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Output size: ";
  for (size_t i = 0; i < num_output_dims; ++i)
  {
    std::cout << output_tensor_info.GetShape()[i] << " ";
  }
  std::cout << std::endl;

  return 0;
  */

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
  if (binding_)
  {
    binding_.ClearBoundInputs();
    binding_.ClearBoundOutputs();
  }
};

void OnnxContainer::SetUpEp(std::string execution_provider)
{
  if (execution_provider == "CPU")
  {
    execution_provider_ = ExecutionProviders::CPU;
    SetUpCpu();
  }
  else if (execution_provider == "CUDA")
  {
    execution_provider_ = ExecutionProviders::CUDA;
    SetUpCuda();
  }
  else if (execution_provider == "TensorRT")
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
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
  input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_arr_,
      input_size_, input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, output_arr_,
      output_size_, output_shape_.data(), output_shape_.size());

  session_ = Ort::Session(env_, model_path_, session_options_);
};

void OnnxContainer::SetUpCuda()
{
  auto const& api = Ort::GetApi();
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  std::unique_ptr<OrtCUDAProviderOptionsV2,
      decltype(api.ReleaseCUDAProviderOptions)>
      rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  std::vector<const char*> keys{"enable_cuda_graph"};
  std::vector<const char*> values{"1"};
  api.UpdateCUDAProviderOptions(
      rel_cuda_options.get(), keys.data(), values.data(), 1);
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
  std::vector<const char*> keys{"trt_cuda_graph_enable"};
  std::vector<const char*> values{"1"};
  api.UpdateTensorRTProviderOptions(
      rel_trt_options.get(), keys.data(), values.data(), keys.size());
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
      "Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session_, info_cuda);

  auto input_data_ = cuda_allocator.GetAllocation(input_size_ * sizeof(float));
  auto output_data_ =
      cuda_allocator.GetAllocation(output_size_ * sizeof(float));
  cudaMemcpy(input_data_.get(), input_arr_, sizeof(float) * input_size_,
      cudaMemcpyHostToDevice);
  cudaMemcpy(output_data_.get(), output_arr_, sizeof(float) * output_size_,
      cudaMemcpyHostToDevice);
  input_tensor_ = Ort::Value::CreateTensor(info_cuda,
      reinterpret_cast<float*>(input_data_.get()), input_size_,
      input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor(info_cuda,
      reinterpret_cast<float*>(output_data_.get()), output_size_,
      output_shape_.data(), output_shape_.size());

  output_device_data_ = reinterpret_cast<float*>(output_data_.get());

  binding_ = Ort::IoBinding(session_);
  binding_.BindInput(input_node_, input_tensor_);
  binding_.BindOutput(output_node_, output_tensor_);

  // One regular run for necessary memory allocation and graph capturing
  session_.Run(run_options_, binding_);

  // Initialize output data
  cudaMemset(output_data_.get(), 0., sizeof(float) * output_size_);
};

void OnnxContainer::Run()
{
  if (binding_)
  {
    session_.Run(run_options_, binding_);
    PullOutput();
  }
}

void OnnxContainer::Run(
    const char* const* input_names, const char* const* output_names)
{
  Ort::RunOptions run_options;
  session_.Run(run_options, input_names, &input_tensor_, 1, output_names,
      &output_tensor_, 1);
}

void OnnxContainer::PullOutput()
{
  cudaMemcpy(output_arr_, output_device_data_, sizeof(float) * output_size_,
      cudaMemcpyDefault);
}
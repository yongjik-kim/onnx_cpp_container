#include "onnx_inference.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
// #include <cuda_provider_factory.h>

class OnnxContainer
{
public:
  OnnxContainer(std::string model_path, int input_h, int input_w, int input_c,
                int output_h, int output_w, int output_c);
  ~OnnxContainer();
  void Run(std::vector<float> &vResults);

  std::vector<float> results_;
  std::vector<float> input_image_;

private:
  Ort::Env env;
  Ort::Session *session_;

  Ort::Value input_tensor_{nullptr};
  std::vector<int64_t> input_shape_;

  Ort::Value output_tensor_{nullptr};
  std::vector<int64_t> output_shape_;
};

OnnxContainer::OnnxContainer(std::string model_path, int input_h, int input_w, int input_c,
                             int output_h, int output_w, int output_c)
{
  wchar_t *wPath = new wchar_t[model_path.length() + 1];
  std::copy(model_path.begin(), model_path.end(), model_path);
  wPath[sPath.length()] = 0;

  session_ = new Ort::Session(env, wPath, Ort::SessionOptions{nullptr});
  delete[] wPath;

  const int batch(1), ;
  const int channel_ = nInputC;
  const int width_ = nInputWidth;
  const int height_ = nInputHeight;

  input_image_.assign(width_ * height_ * channel_, 0.0);
  results_.assign(nOutputDims, 0.0);

  input_shape_ = {batch, input_c, input_h, input_w}; // in NCHW format
  output_shape_ = {batch, output_c, output_h, output_w}; // in NCHW format

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  input_tensor_ =
      Ort::Value::CreateTensor<float>(memory_info, input_image_.data(),
                                      input_image_.size(), input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(),
                                                   results_.size(), output_shape_.data(), output_shape_.size());
}

onnx_module::onnx_module(std::string sModelPath, int nInputC, int nInputWidth,
                         int nInputHeight, int nOutputC, int nOutputWidth, int nOutputHeight)
{
  std::string sPath = sModelPath;
  wchar_t *wPath = new wchar_t[sPath.length() + 1];
  std::copy(sPath.begin(), sPath.end(), wPath);
  wPath[sPath.length()] = 0;

  session_ = new Ort::Session(env, wPath, Ort::SessionOptions{nullptr});
  delete[] wPath;

  const int batch_ = 1;

  const int channel_in = nInputC;
  const int width_in = nInputWidth;
  const int height_in = nInputHeight;

  const int channel_out = nOutputC;
  const int width_out = nOutputWidth;
  const int height_out = nOutputHeight;

  input_image_.assign(width_in * height_in * channel_in, 0.0);
  results_.assign(nOutputWidth * nOutputHeight * nOutputC, 0.0);

  input_shape_.clear();
  input_shape_.push_back(batch_);
  input_shape_.push_back(channel_in);
  input_shape_.push_back(width_in);
  input_shape_.push_back(height_in);

  output_shape_.clear();
  output_shape_.push_back(batch_);
  output_shape_.push_back(channel_out);
  output_shape_.push_back(width_out);
  output_shape_.push_back(height_out);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  input_tensor_ =
      Ort::Value::CreateTensor<float>(memory_info, input_image_.data(),
                                      input_image_.size(), input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(),
                                                   results_.size(), output_shape_.data(), output_shape_.size());
}

void onnx_module::Run(std::vector<float> &vResults)
{
  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};

  (*session_).Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1,
                  output_names, &output_tensor_, 1);

  vResults.assign(results_.begin(), results_.end());
}
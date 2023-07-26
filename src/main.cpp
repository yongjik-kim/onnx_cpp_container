#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "onnx_inference.h"

DEFINE_string(provider, "CPU", "Execution provider: CPU, CUDA, or TensorRT");
DEFINE_string(image_file, "cat.jpg", "Image file to run inference on");
// DEFINE_string(model_file, "yolo.onnx", "Directory to ONNX Model file");
DEFINE_uint32(device, 0, "Device number to run inference on");

int main(int argc, char* argv[])
{
  // Initializing meta variables
  int64_t const input_c(3), input_h(224), input_w(224), output_l(1000);
  std::vector<int64_t> input_shape{input_c, input_h, input_w},
      output_shape{output_l};

  float* input_arr(new float[input_c * input_h * input_w]);
  float* output_arr(new float[output_l]);

  memset(input_arr, 0., input_c * input_h * input_w * sizeof(float));
  memset(output_arr, 0., output_l);

  // Parsing model, processing input

  // Preprocessing for this particular model requires you to process image
  // mean = 255*[0.485, 0.456, 0.406], std = 255*[0.229, 0.224, 0.225]
  // https://github.com/onnx/models/tree/main/vision/classification/resnet

  cv::Mat image = cv::imread(FLAGS_image_file, cv::IMREAD_COLOR);
  cv::Mat input_mat(input_h, input_w, CV_32FC3, output_arr);

  if (image.empty())
  {
    throw std::runtime_error("Error: Unable to read the image");
    return;
  }

  image.convertTo(image, CV_32FC3, 1.0 / 255.0);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::transpose(image, input_mat);  // Now input_arr has all required values

#ifdef _WIN32
  ORTCHAR_T* model_name = L"resnet152-v2-7.onnx";
#elif
  ORTCHAR_T* model_name = "resnet152-v2-7.onnx";
#endif

  OnnxContainer engine(
      model_name, "CUDA", input_shape, output_shape, input_arr, output_arr);
  engine.Run();

  // Processing output

  delete[] input_arr;
  delete[] output_arr;

  return 0;
};
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "onnx_inference.h"

DEFINE_string(provider, "CUDA", "Execution provider: CPU, CUDA, or TensorRT");
DEFINE_string(image_file, "dog.jpg", "Image file to run inference on");
// DEFINE_string(model_file, "yolo.onnx", "Directory to ONNX Model file");
DEFINE_uint32(device, 0, "Device number to run inference on");

std::vector<std::string> ReadClassFile(const std::string& filename)
{
  std::vector<std::string> labels;
  std::ifstream file(filename);

  if (!file.is_open())
  {
    std::cerr << "Error: Failed to open the class file." << std::endl;
    return labels;
  }

  std::string line;
  while (std::getline(file, line))
  {
    // Get the class name string by stripping the line before the first space
    size_t space_pos = line.find(' ');
    if (space_pos != std::string::npos)
    {
      labels.push_back(line.substr(space_pos + 1));
    }
  }

  file.close();
  return labels;
}

void SoftmaxArray(float* array, int length)
{
  // This implementation is ought to be super slow...
  assert(length > 0);

  // Find the maximum value in the array
  float max_val = array[0];
  for (int i = 1; i < length; i++)
  {
    if (array[i] > max_val)
    {
      max_val = array[i];
    }
  }

  // Compute the softmax function
  float sum_exp = 0.0f;
  for (int i = 0; i < length; i++)
  {
    array[i] = std::exp(array[i] - max_val);
    sum_exp += array[i];
  }

  // Normalize the array using the sum of exponentials
  for (int i = 0; i < length; i++)
  {
    array[i] /= sum_exp;
  }
};

cv::Mat ResizeAndCropCenter(const cv::Mat& input_image, int crop_size)
{
  // Step 1: Resize the image to 256 x 256
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, cv::Size(256, 256));

  // Step 2: Calculate the cropping region
  int top = (resized_image.rows - crop_size) / 2;
  int left = (resized_image.cols - crop_size) / 2;

  // Step 3: Crop the image from the center to 224 x 224
  cv::Rect crop_region(left, top, crop_size, crop_size);
  cv::Mat cropped_image = resized_image(crop_region).clone();

  return cropped_image;
}

std::vector<std::pair<float, int>> SortWithIndexes(const float* array, int size)
{
  std::vector<std::pair<float, int>> indexed_array;
  indexed_array.reserve(2 * size * sizeof(float));
  for (int i = 0; i < size; i++)
  {
    indexed_array.push_back(std::make_pair(array[i], i));
  }

  std::sort(indexed_array.begin(), indexed_array.end(),
      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
      });

  return indexed_array;
}

int main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initializing variables
  int64_t const input_n(1), input_c(3), input_h(224), input_w(224), output_n(1),
      output_l(1000);
  std::vector<int64_t> input_shape{input_n, input_c, input_h, input_w},
      output_shape{output_n, output_l};

  // float* input_arr(new float[input_n * input_c * input_h * input_w]);
  float* output_arr(new float[output_n * output_l]);

  // memset(input_arr, 0., input_n * input_c * input_h * input_w * sizeof(float));
  memset(output_arr, 0., output_n * output_l * sizeof(float));

  // Parsing model

  cv::Scalar mean(123.675, 116.28, 103.53);
  cv::Scalar std(58.395, 57.12, 57.375);

  cv::Mat image_raw = cv::imread(FLAGS_image_file, cv::IMREAD_COLOR);

  if (image_raw.empty())
  {
    throw std::runtime_error("Error: Unable to read the image");
    return 0;
  }

  cv::Mat image = ResizeAndCropCenter(image_raw, 224);
  // Preprocessing

  // Preprocessing for this particular model requires you to process image
  // mean = 255*[0.485, 0.456, 0.406], std = 255*[0.229, 0.224, 0.225]
  // https://github.com/onnx/models/tree/main/vision/classification/resnet

  cv::Mat input_mat(input_h, input_w, CV_32FC3);
  cv::cvtColor(image, input_mat, cv::COLOR_BGR2RGB);
  // cv::transpose(input_mat, input_mat);

  std::vector<cv::Mat> channels;
  cv::split(input_mat, channels);
  for (int i = 0; i < 3; i++)
  {
    cv::Mat channelFloat;
    channels[i].convertTo(channelFloat, CV_32F);
    cv::subtract(channelFloat, mean[i], channelFloat);
    cv::divide(channelFloat, std[i], channelFloat);
    channels[i] = channelFloat;
  }
  cv::Mat norm_mat;
  cv::merge(channels, norm_mat);

#ifdef _WIN32
  ORTCHAR_T* model_name = L"resnet152-v2-7.onnx";
#elif
  ORTCHAR_T* model_name = "resnet152-v2-7.onnx";
#endif

  cv::Mat chw_image = cv::dnn::blobFromImage(
    norm_mat, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F
  );

  int num_classes(1000);

  float* input_arr = reinterpret_cast<float*>(chw_image.data);
  
  const char* input_names[] = {"data"};
  const char* output_names[] = {"resnetv27_dense0_fwd"};

  OnnxContainer engine(
      model_name, input_names[0], output_names[0], FLAGS_provider.c_str(), input_shape, output_shape, input_arr, output_arr);

  if (FLAGS_provider == "CPU")
    engine.Run(input_names, output_names);
  else
    engine.Run();

  SoftmaxArray(output_arr, num_classes);

  // Processing output
  std::vector<std::pair<float, int>> sortedWithIndexes =
      SortWithIndexes(output_arr, num_classes);
  std::vector<std::string> class_dict;
  std::string class_file("synset.txt");
  class_dict = ReadClassFile(class_file);

  for (int i = 0; i < 5; i++)
  {
    std::cout << "Rank " << i + 1 << " class: \""
              << class_dict[sortedWithIndexes[i].second]
              << "\" with probability: " << sortedWithIndexes[i].first
              << std::endl;
  }

  // delete[] input_arr;
  delete[] output_arr;

  return 0;
};
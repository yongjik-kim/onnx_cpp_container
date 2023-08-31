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

DEFINE_string(provider, "CPU", "Execution provider: CPU, CUDA, or TensorRT");
DEFINE_string(image_file, "./yolo_images_examples/resized_output_0031.jpg",
    "Image file to run inference on");
DEFINE_string(class_file, "./coco.data", "Image file to run inference on");
DEFINE_double(obj_thresh, 0.1, "Object score threshold");
// DEFINE_string(model_file, "yolo.onnx", "Directory to ONNX Model file");
DEFINE_uint32(device, 0, "Device number to run inference on");

void show_detection(std::vector<std::string> class_dict, cv::Mat image_raw,
    float* detections, int bbox_nums, float obj_score_thresh = 0.1)
{
  // Loop over each detection to draw the bounding box
  for (int i = 0; i < 6 * bbox_nums; i += 6)
  {
    float obj_score = detections[i + 1];
    if (obj_score < obj_score_thresh)
      continue;
    std::string class_label = class_dict[static_cast<int>(detections[i])];
    float l = detections[i + 2];
    float t = detections[i + 3];
    float r = detections[i + 4];
    float b = detections[i + 5];

    // Convert normalized coordinates to actual pixel values
    int left = static_cast<int>(l * image_raw.cols);
    int top = static_cast<int>(t * image_raw.rows);
    int right = static_cast<int>(r * image_raw.cols);
    int bottom = static_cast<int>(b * image_raw.rows);

    // Draw rectangle around the object
    cv::rectangle(image_raw, cv::Point(left, top), cv::Point(right, bottom),
        cv::Scalar(0, 255, 0), 2);

    // Display the label and confidence score
    std::string label_text =
        class_label + " (" + std::to_string(obj_score) + ")";
    cv::putText(image_raw, label_text, cv::Point(left, top - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }

  // Display the image with detections
  cv::namedWindow("YOLO Detections", cv::WINDOW_NORMAL);
  cv::imshow("YOLO Detections", image_raw);
  cv::waitKey(0);  // Wait for any key press
  cv::destroyAllWindows();
}

std::vector<std::string> ReadCocoClassFile(const std::string& filename)
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
    labels.push_back(line);
  }

  file.close();
  return labels;
}

int main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::string> class_dict;
  class_dict = ReadCocoClassFile(FLAGS_class_file);

  // Initializing variables
  int64_t const input_n(1), input_c(3), input_h(640), input_w(640);
  // Number of bboxes is dynamic, so we have to read output_n from the actual
  // onnx output.
  std::vector<int64_t> input_shape{input_n, input_c, input_h, input_w},
      scale_shape{input_n, 2};
  std::vector<float> input_scale{640.0, 640.0};

  // Parsing model
  cv::Mat image_raw = cv::imread(FLAGS_image_file, cv::IMREAD_COLOR);

  if (image_raw.empty())
  {
    throw std::runtime_error("Error: Unable to read the image");
    return 0;
  }

  // Preprocessing
  // For PP-YOLO-E, normalization is inside onnx model.
  // However, you still need to manually resize.
  cv::Mat input_mat(input_h, input_w, CV_32FC3);
  cv::Mat norm_mat(640, 640, CV_32FC3);
  cv::cvtColor(image_raw, input_mat, cv::COLOR_BGR2RGB);
  cv::resize(input_mat, norm_mat, cv::Size(640, 640));

  // The onnx model is exported from PaddleDetection project
  // Source: https://github.com/PaddlePaddle/PaddleDetection

#ifdef _WIN32
  ORTCHAR_T* model_name = L"ppyoloe_plus_crn_l_80e_coco_w_nms.onnx";
#elif
  ORTCHAR_T* model_name = "ppyoloe_plus_crn_l_80e_coco_w_nms.onnx";
#endif

  cv::Mat chw_image = cv::dnn::blobFromImage(
      norm_mat, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

  float* image_arr = reinterpret_cast<float*>(chw_image.data);

  const std::vector<const char*> input_names = {"image", "scale_factor"};
  const std::vector<const char*> output_names = {
      "multiclass_nms3_0.tmp_0", "multiclass_nms3_0.tmp_2"};

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options = Ort::SessionOptions();
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);  // Change to ORT_ENABLE_BASIC
                                                  // if gone wrong

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
  auto input_tensor_image = Ort::Value::CreateTensor<float>(memory_info,
      image_arr, 1 * 3 * 640 * 640, input_shape.data(), input_shape.size());
  auto input_tensor_scale =
      Ort::Value::CreateTensor<float>(memory_info, input_scale.data(),
          input_scale.size(), scale_shape.data(), scale_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor_image));
  input_tensors.push_back(std::move(input_tensor_scale));

  Ort::Session onnx_session = Ort::Session(env, model_name, session_options);

  auto output_tensors = onnx_session.Run(Ort::RunOptions{nullptr},
      input_names.data(), input_tensors.data(), input_names.size(),
      output_names.data(), output_names.size());
  // Extract output tensor
  Ort::Value& output1 = output_tensors[0];  // We only take the first output

  // Get output shape
  auto tensor_info = output1.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_shape = tensor_info.GetShape();

  // Print output shape
  std::cout << "Output shape: ";
  for (const auto& dim : output_shape)
  {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Print raw output values for examination
  // Here, we use onnx with NMS implemented, so output shape is dynamic.
  // This also means that class score is not visible.
  // (onnx can be built with NMS, tensorrt can't natively be built with NMS.)

  float* floatarr = output1.GetTensorMutableData<float>();
  std::cout << "Output values: ";
  for (int i = 0; i < tensor_info.GetElementCount(); ++i)
  {
    if (i % 6 == 0)
      std::cout << std::endl;
    // The output format will be [class_score, obj_score, l, t, r, b]
    std::cout << floatarr[i] << " ";
  }
  std::cout << std::endl;

  int64_t bbox_nums = output_shape[0];

  show_detection(class_dict, image_raw, floatarr, static_cast<int>(bbox_nums), static_cast<float>(FLAGS_obj_thresh));

  return 0;
};
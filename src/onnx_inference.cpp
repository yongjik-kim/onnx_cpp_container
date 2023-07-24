#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
// #include <cuda_provider_factory.h>

class OnnxContainer
{
 public:
  OnnxContainer(std::string model_path, int input_h, int input_w, int input_c,
      int output_dims);
  OnnxContainer(std::string model_path, int input_h, int input_w, int input_c,
      int output_h, int output_w, int output_c);
  ~OnnxContainer();
  void Run(std::vector<float>& vResults);

  std::vector<float> results_;
  std::vector<float> input_image_;

 private:
  Ort::Env env;
  Ort::Session* session_;

  Ort::Value input_tensor_{nullptr};
  std::vector<int64_t> input_shape_;

  Ort::Value output_tensor_{nullptr};
  std::vector<int64_t> output_shape_;
};
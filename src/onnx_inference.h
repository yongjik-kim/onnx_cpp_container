#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>
// #include <cuda_provider_factory.h>

#ifdef _WIN32
#define ORTCHAR_T wchar_t
#else
#define ORTCHAR_T char
#endif

enum class ExecutionProviders
{
  CPU,
  CUDA,
  TENSORRT,
};

class OnnxContainer
{
 public:
  OnnxContainer(ORTCHAR_T* model_path, std::string execution_provider,
      std::vector<int64_t> input_shape, std::vector<int64_t> output_shape,
      float* input_arr, float* output_arr);
  ~OnnxContainer();
  void Run();
  void OnnxContainer::Run(
      const char* const* input_names, const char* const* output_names);
 private:
  void SetUpCpu();
  void SetUpCuda();
  void SetUpTensorrt();
  void SetUpGpuIoBindings();
  void SetUpEp(std::string execution_provider);

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;

  Ort::RunOptions run_options_;
  Ort::IoBinding binding_;

  Ort::Value input_tensor_{nullptr}, output_tensor_{nullptr};
  std::vector<int64_t> input_shape_, output_shape_;

  ExecutionProviders execution_provider_;

  size_t input_size_, output_size_;
  float *input_arr_, *output_arr_;

  ORTCHAR_T* model_path_;
};
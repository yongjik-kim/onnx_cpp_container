# (WIP) onnxruntime container w/ IObinding on C++

## Introduction

Despite `.onnx` models having clear benefits in serving deep learning models on C++ environment, most guides on running inference with `.onnx` models have focused on Python onnx runtime. Still, there are helpful resources regarding C++ implementation:

- [Microsoft's official onnxruntime inference examples on C/C++](https://github.com/microsoft/onnxruntime-inference-examples)

In this repo, I provide a simple but robust container `OnnxContainer` which you can hopefully employ on your production code. Supported execution providers are CPU, CUDA and TensorRT.

## What is special about this repo

1. IObinding implemented to properly utilize device memory
2. TBA

## Further readings

TBA
#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> vsl_intersection_cuda(
    torch::Tensor data1,
    torch::Tensor indexes1,
    torch::Tensor data2,
    torch::Tensor indexes2);

std::vector<torch::Tensor> vsl_intersection(
    torch::Tensor data1,
    torch::Tensor indexes1,
    torch::Tensor data2,
    torch::Tensor indexes2) {
  CHECK_INPUT(data1);
  CHECK_INPUT(indexes1);
  CHECK_INPUT(data2);
  CHECK_INPUT(indexes2);
  return vsl_intersection_cuda(data1, indexes1, data2, indexes2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vsl_intersection", &vsl_intersection, "VSL intersection (CUDA)");
}
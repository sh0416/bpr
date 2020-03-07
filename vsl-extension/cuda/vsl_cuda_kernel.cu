#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void vsl_intersection_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data2,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes2) {
    
}

std::vector<torch::Tensor> vsl_intersection_cuda(
    torch::Tensor data1,
    torch::Tensor indexes1,
    torch::Tensor data2,
    torch::Tensor indexes2) {
    
    int blocks = 10;
    int threads = 1024;
    AT_DISPATCH_FLOATING_TYPES(data1.type(), "vsl_intersection_cuda_kernel", ([&] {
        vsl_intersection_cuda_kernel<scalar_t><<<blocks, threads>>>(
            data1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            indexes1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            data2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            indexes2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    return {};
}
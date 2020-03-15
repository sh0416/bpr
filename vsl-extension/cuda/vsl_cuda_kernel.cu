#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


template <typename scalar_t>
__global__ void vsl_mask(
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data2,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes2,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mask) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    for(int data1_idx=idx;
            data1_idx<data1.size(0);
            data1_idx+=gridDim.x*blockDim.x) {
        int indexes_idx = -1;
        for(int j=0; j<indexes1.size(0)-1; j++)
            if(indexes1[j]<=data1_idx && data1_idx<indexes1[j+1]) {
                indexes_idx = j;
                break;
            }
        if (0<=indexes_idx && indexes_idx<indexes1.size(0)-1)
            for(int data2_idx=indexes2[indexes_idx];
                    data2_idx<indexes2[indexes_idx+1];
                    data2_idx++) {
                if(data1[data1_idx] == data2[data2_idx]) {
                    mask[data1_idx] = 1;
                    break;
                }
            }
    }
}


template <typename scalar_t>
__global__ void prefix_scan(
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out) {
	// WARN: EXCLUSIVE SCAN NOT INCLUDING LAST COMPONENT!!!
    extern __shared__ scalar_t temp[2048];
    int thid = threadIdx.x;
    int offset = 1;
    
    int ai = thid;
    int bi = thid + (in.size(0)/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = in[ai];
    temp[bi + bankOffsetB] = in[bi];

    for (int d=in.size(0)>>1; d>0; d>>=1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0)
        temp[in.size(0) - 1 + CONFLICT_FREE_OFFSET(in.size(0) - 1)] = 0;
    for (int d = 1; d<in.size(0); d*=2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            scalar_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    out[ai + bankOffsetA] = temp[ai + bankOffsetA];
	out[bi + bankOffsetB] = temp[bi + bankOffsetB];
}


/*
template <typename scalar_t>
__global__ void prefix_scan(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t val = 0;
	for (int i=0; i<in.size(0); i++) {
		val += in[i];
		out[i] = val;
	}
}
*/


template <typename scalar_t>
__global__ void create_vsl_result(
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_data,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_indexes,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> sum,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out_data,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out_indexes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i=idx; i<in_data.size(0); i+=gridDim.x*blockDim.x)
        if (mask[i] == 1)
            out_data[sum[i]] = in_data[i];
    for (unsigned int i=idx; i<in_indexes.size(0)-1; i+=gridDim.x*blockDim.x)
        out_indexes[i] = sum[in_indexes[i]];
	if (idx == 0)
		out_indexes[in_indexes.size(0)-1] = sum[sum.size(0)-1] + mask[mask.size(0)-1];
}


std::vector<torch::Tensor> vsl_intersection_cuda(
    torch::Tensor data1,
    torch::Tensor indexes1,
    torch::Tensor data2,
    torch::Tensor indexes2) {
    
    // Create result tensor
    auto mask = torch::zeros_like(data1);
    auto sum  = torch::zeros_like(data1);
    auto result_data = torch::empty_like(data1);
    auto result_indexes = torch::zeros_like(indexes1);

    // Caclulate result_data and result_indexes
    int blocks = 1;
    int threads = 1024;
    //AT_DISPATCH_LONG_TYPES(data1.type(), "vsl_intersection_cuda_kernel", ([&] {
    vsl_mask<long><<<blocks, threads>>>(
        data1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        data2.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes2.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    //}));
    prefix_scan<long><<<1, 1024>>>(
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        sum.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    create_vsl_result<long><<<blocks, threads>>>(
        data1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        sum.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        result_data.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        result_indexes.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    return {result_data, result_indexes, mask, sum};
}

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


template <typename scalar_t>
__global__ void vsl_mask(
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes1,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> data2,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> indexes2,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mask) {
    //int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    extern __shared__ scalar_t data2_shared[32];

    for(int element_idx=blockIdx.x;
            element_idx<indexes1.size(0)-1;
            element_idx+=gridDim.x) {
        // Load data1
        for(int data1_offset=indexes1[element_idx];
                data1_offset<(indexes1[element_idx+1]+(blockDim.x-indexes1[element_idx]%blockDim.x));
                data1_offset+=blockDim.x) {
            int data1_val;
            int mask_val;
            if (data1_offset + threadIdx.x < indexes1[element_idx+1]) {
                data1_val = data1[data1_offset+threadIdx.x];
                mask_val = 0;
            }
            for(int data2_offset=indexes2[element_idx];
                    data2_offset<(indexes2[element_idx+1]+(blockDim.x-indexes2[element_idx]%blockDim.x));
                    data2_offset+=blockDim.x) {
                // Load data2 to shared memory
                if (data2_offset + threadIdx.x < indexes2[element_idx+1])
                    data2_shared[threadIdx.x] = data2[data2_offset+threadIdx.x];
                __syncthreads();
                // Compare
                if (data1_offset + threadIdx.x < indexes1[element_idx+1])
                    for(int i=0; (i<blockDim.x && data2_offset+i<indexes2[element_idx+1]); i++)
                        if(data1_val == data2_shared[i])
                            mask_val = 1;
                __syncthreads();
            }
            if (data1_offset + threadIdx.x < indexes1[element_idx+1])
                mask[data1_offset + threadIdx.x] = mask_val;
        }
    }
}


template <typename scalar_t>
__global__ void prefix_scan2(
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out) {
	// WARN: EXCLUSIVE SCAN NOT INCLUDING LAST COMPONENT!!!
    extern __shared__ scalar_t temp[2048];
    int thid = threadIdx.x;

    scalar_t prev_sum = 0, last;
    
    for (int start=0; start<in.size(0); start+=2048) {
        int n = start+2048 < in.size(0) ? 2048 : in.size(0)-start;
        int offset = 1;
        
        temp[2*thid] = 0;
        temp[2*thid+1] = 0;
        __syncthreads();
        int ai = thid;
        int bi = thid + (n/2);
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        temp[ai + bankOffsetA] = in[start + ai];
        temp[bi + bankOffsetB] = in[start + bi];
        __syncthreads();
        
        last = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        for (int d=n>>1; d>0; d>>=1) {
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
            temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        for (int d = 1; d<n; d*=2) {
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
        out[start + ai] = prev_sum + temp[ai + bankOffsetA];
        out[start + bi] = prev_sum + temp[bi + bankOffsetB];
        __syncthreads();
        prev_sum += temp[n-1 + CONFLICT_FREE_OFFSET(n-1)] + last;
        __syncthreads();
    }
}



template <typename scalar_t>
__global__ void prefix_scan(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out) {
	scalar_t val = 0;
	for (int i=0; i<in.size(0); i++) {
		out[i] = val;
		val += in[i];
	}
}


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
    vsl_mask<long><<<8192, 32>>>(
        data1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        data2.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes2.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    //}));
    /*
    prefix_scan<long><<<1, 1>>>(
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        sum.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
        */
    prefix_scan2<long><<<1, 1024>>>(
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        sum.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    create_vsl_result<long><<<1024, 1024>>>(
        data1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        indexes1.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        mask.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        sum.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        result_data.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        result_indexes.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>());
    return {result_data, result_indexes, mask, sum};
}

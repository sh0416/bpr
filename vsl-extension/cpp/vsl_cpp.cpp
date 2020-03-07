#include <torch/extension.h>

#include <iostream>
#include <vector>


std::vector<torch::Tensor> vsl_intersection(
    torch::Tensor data1,
    torch::Tensor indexes1,
    torch::Tensor data2,
    torch::Tensor indexes2) {
    
    // Option for tensor
    auto options = torch::TensorOptions().dtype(torch::kInt64)
                                         .layout(torch::kStrided)
                                         .device(torch::kCPU)
                                         .requires_grad(false);

    // Calculate result indexes
    auto result_indexes = torch::zeros({indexes1.size(0)}, options);
    auto indexes1_a = indexes1.accessor<int64_t, 1>();
    auto indexes2_a = indexes2.accessor<int64_t, 1>();

    // Set result_indexes with worst case
    int64_t tmp = 0;
    for(int i=0; i<result_indexes.size(0)-1; i++) {
        int64_t element1_size = indexes1_a[i+1] - indexes1_a[i];
        int64_t element2_size = indexes2_a[i+1] - indexes2_a[i];
        tmp += std::max(element1_size, element2_size);
        result_indexes[i+1] = tmp;
    }

    // Calculate result_data and change result_indexes with real value
    int64_t idx = 0;
    auto result_data = torch::empty(tmp, options);
    auto data1_a = data1.accessor<int64_t, 1>();
    auto data2_a = data2.accessor<int64_t, 1>();
    for(int i=0; i<result_indexes.size(0)-1; i++) {
        int64_t element1_size = indexes1_a[i+1] - indexes1_a[i];
        int64_t element2_size = indexes2_a[i+1] - indexes2_a[i];
        for(int j=0; j<element1_size; j++) {
            for(int k=0; k<element2_size; k++){
                if(data1_a[indexes1_a[i]+j] == data2_a[indexes2_a[i]+k]){
                    result_data[idx] = data1_a[indexes1_a[i]+j];
                    idx += 1;
                    break;
                }
            }
        }
        result_indexes[i+1] = idx;
    }
    return {result_data, result_indexes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vsl_intersection", &vsl_intersection, "VSL intersection");
}
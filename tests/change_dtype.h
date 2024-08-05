#include <cassert>
#include <memory>
#include "tensor.h"


void test_change_dtype() { 
    std::vector<int> shape = {3, 3};
    Tensor<INT32> tensor(shape);
    for (int i = 0; i < 9; ++i) {
        tensor.set({i / 3, i % 3}, i);
    }

    // Change the tensor's data type to FLOAT32
    auto new_tensor = tensor.change_dtype<FLOAT32>();
    
    // Verify the values in the new tensor
    for (int i = 0; i < 9; ++i) {
        assert(new_tensor->get({i / 3, i % 3}) == static_cast<float>(i)); // Compare to float
    }

    // Change the tensor's data type to UINT8
    auto new_tensor_uint8 = tensor.change_dtype<UINT8>();
    
    // Verify the values in the UINT8 tensor
    for (int i = 0; i < 9; ++i) {
        assert(new_tensor_uint8->get({i / 3, i % 3}) == static_cast<uint8_t>(i)); // Compare to uint8_t
    }

    // Change the tensor's data type to INT8
    auto new_tensor_int8 = tensor.change_dtype<INT8>();
    
    // Verify the values in the INT8 tensor
    for (int i = 0; i < 9; ++i) {
        assert(new_tensor_int8->get({i / 3, i % 3}) == static_cast<int8_t>(i)); // Compare to int8_t
    }

    // Change the tensor's data type to UINT32
    auto new_tensor_uint32 = tensor.change_dtype<UINT32>();
    
    // Verify the values in the UINT32 tensor
    for (int i = 0; i < 9; ++i) {
        assert(new_tensor_uint32->get({i / 3, i % 3}) == static_cast<uint32_t>(i)); // Compare to uint32_t
    }
    std::cout << "All tests passed!" << std::endl;
}

#ifndef SET_SLICE_H
#define SET_SLICE_H

#include "tensor.h"
#include <iostream>
#include <vector>

void print_tensor(const Tensor<UINT32>& tensor) {
    std::cout << tensor << std::endl;  
}

void test_set_slice() {
    //some tests not passing
    // Test with 3D tensor
    std::vector<int> shape3d{1, 3, 3};
    std::vector<uint32_t> data3d{1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor<UINT32> test3d(data3d, shape3d);
    std::cout << "Original 3D tensor:" << std::endl;
    print_tensor(test3d);

    // Test 1: Basic slice with valid data
    std::vector<uint32_t> values3d{10, 11, 12, 13, 14, 15};
    std::vector<int> start_indices3d{0, 0, 0};
    std::vector<int> end_indices3d{1, 2, 3}; // Note: This should match the tensor dimensions
    test3d.set_slice(start_indices3d, end_indices3d, values3d);
    std::cout << "Test 1: Basic slice (3D)" << std::endl;
    print_tensor(test3d);

    // Test 2: Overwriting a slice with new data
    std::vector<uint32_t> values3d_2{20, 21, 22, 23, 24, 25};
    test3d.set_slice(start_indices3d, end_indices3d, values3d_2);
    std::cout << "Test 2: Overwriting slice (3D)" << std::endl;
    print_tensor(test3d);

    // Test with 4D tensor
    std::vector<int> shape4d{2, 2, 2, 2};
    std::vector<uint32_t> data4d{
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    };
    Tensor<UINT32> test4d(data4d, shape4d);
    std::cout << "Original 4D tensor:" << std::endl;
    print_tensor(test4d);

    // Test 1: Basic slice with valid data
    std::vector<uint32_t> values4d{100, 101, 102, 103};
    std::vector<int> start_indices4d{0, 0, 0, 0};
    std::vector<int> end_indices4d{1, 1, -1, -1}; // Note: This should match the tensor dimensions
    Tensor<UINT32> sliced_up = test4d.get_slice(start_indices4d, end_indices4d); 
    std::cout<<"SLICED: "<<sliced_up<<std::endl;
    test4d.set_slice(start_indices4d, end_indices4d, values4d);
    std::cout << "Test 1: Basic slice (4D)" << std::endl;
    print_tensor(test4d);

    // Test 2: Overwriting a slice with new data
    std::vector<uint32_t> values4d_2{200, 201, 202, 203, 204, 205, 206, 207};
    test4d.set_slice(start_indices4d, end_indices4d, values4d_2);
    std::cout << "Test 2: Overwriting slice (4D)" << std::endl;
    print_tensor(test4d);

    // Test 3: Edge case with invalid indices (should throw an error)
    try {
        std::vector<uint32_t> invalid_values4d{300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315};
        std::vector<int> invalid_start4d{0, 0, 0, 0};
        std::vector<int> invalid_end4d{3, 3, 3, 3}; // Out of bounds
        test4d.set_slice(invalid_start4d, invalid_end4d, invalid_values4d);
        std::cout << "Test 3: Invalid indices slice (4D) (should throw an error)" << std::endl;
        print_tensor(test4d);
    } catch (const std::runtime_error& e) {
        std::cout << "Test 3: Caught expected runtime error (4D): " << e.what() << std::endl;
    }
}
#endif 

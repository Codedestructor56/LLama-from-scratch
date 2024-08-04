#pragma once

#include <iostream>
#include "tensor.h"


void print_tensor(const Tensor<UINT8>& tensor) {
    std::cout << tensor << std::endl;
}

void test_get_slice() {
    // Test with 3D tensor
    std::vector<int> shape3d{1, 3, 3};
    std::vector<int> data3d{1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor<UINT8> test3d = Tensor<UINT8>(data3d, shape3d);
    std::cout << "Original 3D tensor:" << std::endl;
    print_tensor(test3d);

    // Test 1: Basic slice with default traversal and stride
    std::vector<int> start_indices3d{0, 0, 0};
    std::vector<int> end_indices3d{-1, 2, -1};
    Tensor<UINT8> slice1_3d = test3d.get_slice(start_indices3d, end_indices3d, {1, 1, 1});
    std::cout << "Test 1: Basic slice (3D)" << std::endl;
    print_tensor(slice1_3d);

    // Test 3: Slice with stride
    std::vector<int> stride3d{1, 2, 2};
    Tensor<UINT8> slice3_3d = test3d.get_slice(start_indices3d, end_indices3d, stride3d);
    std::cout << "Test 3: Slice with stride (3D)" << std::endl;
    print_tensor(slice3_3d);

    // Test 4: Slice with -1 end index
    std::vector<int> end_indices_full3d{-1, -1, -1};
    Tensor<UINT8> slice4_3d = test3d.get_slice(start_indices3d, end_indices_full3d);
    std::cout << "Test 4: Slice with -1 end index (3D)" << std::endl;
    print_tensor(slice4_3d);

    // Test 5: Mixed slice with different start, end, and stride
    std::vector<int> mixed_start3d{0, 1, 0};
    std::vector<int> mixed_end3d{1, 2, -1};
    std::vector<int> mixed_stride3d{1, 1, 1};
    Tensor<UINT8> slice5_3d = test3d.get_slice(mixed_start3d, mixed_end3d, mixed_stride3d);
    std::cout << "Test 5: Mixed slice with different start, end, and stride (3D)" << std::endl;
    print_tensor(slice5_3d);

    // Test 6: Edge case with invalid indices (should throw an error)
    try {
        std::vector<int> invalid_start3d{0, 0, 0};
        std::vector<int> invalid_end3d{4, 4, 4}; // Out of bounds
        Tensor<UINT8> slice6_3d = test3d.get_slice(invalid_start3d, invalid_end3d);
        std::cout << "Test 6: Invalid indices slice (3D) (should throw an error)" << std::endl;
        print_tensor(slice6_3d);
    } catch (const std::runtime_error& e) {
        std::cout << "Test 6: Caught expected runtime error (3D): " << e.what() << std::endl;
    }

    // Test with 4D tensor
    std::vector<int> shape4d{2, 2, 2, 2};
    std::vector<int> data4d{
        1, 2, 3, 4, 5, 6, 7, 8, 
        9, 10, 11, 12, 13, 14, 15, 16
    };
    Tensor<UINT8> test4d = Tensor<UINT8>(data4d, shape4d);
    std::cout << "Original 4D tensor:" << std::endl;
    print_tensor(test4d);

    // Test 1: Basic slice with default traversal and stride
    std::vector<int> start_indices4d{0, 0, 0, 0};
    std::vector<int> end_indices4d{2, 2, 2, 2};
    Tensor<UINT8> slice1_4d = test4d.get_slice(start_indices4d, end_indices4d, {1, 1, 1, 1});
    std::cout << "Test 1: Basic slice (4D)" << std::endl;
    print_tensor(slice1_4d);

    // Test 3: Slice with stride
    std::vector<int> stride4d{1, 1, 2, 1};
    Tensor<UINT8> slice3_4d = test4d.get_slice(start_indices4d, end_indices4d, stride4d);
    std::cout << "Test 3: Slice with stride (4D)" << std::endl;
    print_tensor(slice3_4d);

    // Test 4: Slice with -1 end index
    std::vector<int> end_indices_full4d{-1, -1, -1, -1};
    Tensor<UINT8> slice4_4d = test4d.get_slice(start_indices4d, end_indices_full4d);
    std::cout << "Test 4: Slice with -1 end index (4D)" << std::endl;
    print_tensor(slice4_4d);

    // Test 5: Mixed slice with different start, end, and stride
    std::vector<int> mixed_start4d{0, 1, 0, 1};
    std::vector<int> mixed_end4d{2, 2, 2, 2};
    std::vector<int> mixed_stride4d{1, 1, 2, 1};
    Tensor<UINT8> slice5_4d = test4d.get_slice(mixed_start4d, mixed_end4d, mixed_stride4d);
    std::cout << "Test 5: Mixed slice with different start, end, and stride (4D)" << std::endl;
    print_tensor(slice5_4d);

    // Test 6: Edge case with invalid indices (should throw an error)
    try {
        std::vector<int> invalid_start4d{0, 0, 0, 0};
        std::vector<int> invalid_end4d{3, 3, 3, 3}; // Out of bounds
        Tensor<UINT8> slice6_4d = test4d.get_slice(invalid_start4d, invalid_end4d);
        std::cout << "Test 6: Invalid indices slice (4D) (should throw an error)" << std::endl;
        print_tensor(slice6_4d);
    } catch (const std::runtime_error& e) {
        std::cout << "Test 6: Caught expected runtime error (4D): " << e.what() << std::endl;
    }
}


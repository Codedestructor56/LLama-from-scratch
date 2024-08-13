#include <iostream>
#include <vector>
#include <cassert>
#include "tensor.h"

void testAtomicMulTensor() {
    // Test 1: Simple multiplication
    std::vector<int> shape = {4};
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {2.0f, 3.0f, 4.0f, 5.0f};
    
    Tensor<FLOAT32> tensor1(data1.data(), shape);
    Tensor<FLOAT32> tensor2(data2.data(), shape);
    
    atomicMulTensor(tensor1, tensor2);
    
    std::cout<<tensor1<<std::endl;
    std::vector<float> expected1 = {2.0f, 6.0f, 12.0f, 20.0f};
    for (int i = 0; i < expected1.size(); ++i) {
        assert(tensor1.data()[i] == expected1[i]);
    }

    std::cout << "Test 1 passed." << std::endl;
    
    // Test 2: All ones multiplication
    std::vector<float> data3 = {1.0f, 1.0f, 1.0f, 1.0f};
    
    Tensor<FLOAT32> tensor3(data3.data(), shape);
    
    atomicMulTensor(tensor1, tensor3);
    
    std::vector<float> expected2 = {2.0f, 6.0f, 12.0f, 20.0f};
    for (int i = 0; i < expected2.size(); ++i) {
        assert(tensor1.data()[i] == expected2[i]);
    }

    std::cout << "Test 2 passed." << std::endl;

    // Test 3: Multiplication by zero
    std::vector<float> data4 = {0.0f, 0.0f, 0.0f, 0.0f};
    
    Tensor<FLOAT32> tensor4(data4.data(), shape);
    
    atomicMulTensor(tensor1, tensor4);
    
    std::vector<float> expected3 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < expected3.size(); ++i) {
        assert(tensor1.data()[i] == expected3[i]);
    }

    std::cout << "Test 3 passed." << std::endl;
    
    // Test 4: Different shapes (should fail)
    std::vector<int> shape2 = {2, 2};
    std::vector<float> data5 = {2.0f, 2.0f, 2.0f, 2.0f};
    
    Tensor<FLOAT32> tensor5(data5.data(), shape2);
    
    try {
        atomicMulTensor(tensor1, tensor5);
        assert(false); // Should not reach here
    } catch (const std::runtime_error& e) {
        std::cout << "Test 4 passed." << std::endl;
    }

    // Test 5: Large tensor multiplication
    std::vector<int> shape3 = {1000};
    std::vector<float> data6(1000, 2.0f);
    std::vector<float> data7(1000, 3.0f);
    
    Tensor<FLOAT32> tensor6(data6.data(), shape3);
    Tensor<FLOAT32> tensor7(data7.data(), shape3);
    
    atomicMulTensor(tensor6, tensor7);
    
    for (int i = 0; i < 1000; ++i) {
        assert(tensor6.data()[i] == 6.0f);
    }

    std::cout << "Test 5 passed." << std::endl;

    // Test 6: Identity multiplication
    std::vector<float> data8 = {1.0f, 1.0f, 1.0f, 1.0f};
    
    Tensor<FLOAT32> tensor8(data8.data(), shape);
    
    atomicMulTensor(tensor1, tensor8);
    
    std::vector<float> expected4 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < expected4.size(); ++i) {
        assert(tensor1.data()[i] == expected4[i]);
    }

    std::cout << "Test 6 passed." << std::endl;
}

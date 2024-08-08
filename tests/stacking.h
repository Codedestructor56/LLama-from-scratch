#include <iostream>
#include <vector>
#include <cassert>
#include "tensor.h"

void test_stack() {
    // Define some test tensors
    Tensor<FLOAT32> tensor1({2, 3});
    Tensor<FLOAT32> tensor2({2, 3});
    Tensor<FLOAT32> tensor3({2, 2});
    Tensor<FLOAT32> tensor4({3, 3});

    // Initialize tensor1 and tensor2 with some values
    for (int i = 0; i < 6; ++i) {
        tensor1.data()[i] = static_cast<float>(i + 1);
        tensor2.data()[i] = static_cast<float>(i + 7);
    }

    // Test 1: Valid hstack of two tensors with the same shape
    auto hstack_result1 = hstack(tensor1, tensor2);
    std::cout << "HStack Test 1:\n" << hstack_result1 << std::endl;

    // Test 2: Valid vstack of two tensors with the same shape
    auto vstack_result1 = vstack(tensor1, tensor2);
    std::cout << "VStack Test 2:\n" << vstack_result1 << std::endl;

    // Test 3: hstack with a list of tensors
    auto hstack_result2 = hstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor2, tensor1});
    std::cout << "HStack Test 3:\n" << hstack_result2 << std::endl;

    // Test 4: vstack with a list of tensors
    auto vstack_result2 = vstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor2, tensor1});
    std::cout << "VStack Test 4:\n" << vstack_result2 << std::endl;

    // Test 5: hstack with mismatched column size (should throw an error)
    try {
        auto hstack_result3 = hstack(tensor1, tensor3);
        std::cout << "HStack Test 5: (Expected Failure) \n" << hstack_result3 << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "HStack Test 5: Caught expected error - " << e.what() << std::endl;
    }

    // Test 6: vstack with mismatched row size (should throw an error)
    try {
        auto vstack_result3 = vstack(tensor1, tensor4);
        std::cout << "VStack Test 6: (Expected Failure) \n" << vstack_result3 << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "VStack Test 6: Caught expected error - " << e.what() << std::endl;
    }

    // Test 7: Valid hstack of three tensors with the same shape
    auto hstack_result4 = hstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor2, tensor1});
    std::cout << "HStack Test 7:\n" << hstack_result4 << std::endl;

    // Test 8: Valid vstack of three tensors with the same shape
    auto vstack_result4 = vstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor2, tensor1});
    std::cout << "VStack Test 8:\n" << vstack_result4 << std::endl;

    // Test 9: hstack with different shapes that align correctly (should work)
    Tensor<FLOAT32> tensor5({2, 1});
    tensor5.data()[0] = 9.0f;
    tensor5.data()[1] = 10.0f;
    auto hstack_result5 = hstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor5});
    std::cout << "HStack Test 9:\n" << hstack_result5 << std::endl;

    // Test 10: vstack with different shapes that align correctly (should work)
    Tensor<FLOAT32> tensor6({1, 3});
    tensor6.data()[0] = 11.0f;
    tensor6.data()[1] = 12.0f;
    tensor6.data()[2] = 13.0f;
    auto vstack_result5 = vstack(std::vector<Tensor<FLOAT32>>{tensor1, tensor6});
    std::cout << "VStack Test 10:\n" << vstack_result5 << std::endl;

    std::cout<<"\nAll tests passed!"<<std::endl;
}


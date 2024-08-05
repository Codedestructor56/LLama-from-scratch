#include <iostream>
#include <vector>
#include <memory>
#include <variant>
#include "tensor.h"

void generate_test_cases() {
    //operations correct, types of results are incorrect
    // Define shapes for tensors
    std::vector<int> shape1 = {2, 2};
    std::vector<int> shape2 = {3, 3};

    // Define data for different tensor types
    std::vector<float> float_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<uint16_t> float16_data = {1, 2, 3, 4};
    std::vector<int32_t> int32_data = {1, 2, 3, 4};
    std::vector<uint32_t> uint32_data = {1, 2, 3, 4};
    std::vector<int8_t> int8_data = {1, 2, 3, 4};
    std::vector<uint8_t> uint8_data = {1, 2, 3, 4};

    // Create tensors of different types
    auto tensor_float32 = std::make_shared<Tensor<FLOAT32>>(float_data, shape1);
    auto tensor_float16 = std::make_shared<Tensor<FLOAT16>>(float16_data, shape1);
    auto tensor_int32 = std::make_shared<Tensor<INT32>>(int32_data, shape1);
    auto tensor_uint32 = std::make_shared<Tensor<UINT32>>(uint32_data, shape1);
    auto tensor_int8 = std::make_shared<Tensor<INT8>>(int8_data, shape1);
    auto tensor_uint8 = std::make_shared<Tensor<UINT8>>(uint8_data, shape1);

    // Store tensors in a variant
    TensorVariant variant_float32 = tensor_float32;
    TensorVariant variant_float16 = tensor_float16;
    TensorVariant variant_int32 = tensor_int32;
    TensorVariant variant_uint32 = tensor_uint32;
    TensorVariant variant_int8 = tensor_int8;
    TensorVariant variant_uint8 = tensor_uint8;

    // Perform addition, subtraction, and multiplication with tensors of the same type
    auto result_add_float32 = *tensor_float32 + variant_float32;
    auto result_sub_float32 = *tensor_float32 - variant_float32;
    auto result_mul_float32 = *tensor_float32 * variant_float32;

    auto result_add_int32 = *tensor_int32 + variant_int32;
    auto result_sub_int32 = *tensor_int32 - variant_int32;
    auto result_mul_int32 = *tensor_int32 * variant_int32;

    // Print results
    std::cout << "Addition result (float32): " << result_add_float32 << std::endl; 
    std::cout << "Subtraction result (float32): " << result_sub_float32 << std::endl;
    std::cout << "Multiplication result (float32): " << result_mul_float32 << std::endl;

    std::cout << "Addition result (int32): " << result_add_int32 << std::endl;
    std::cout << "Subtraction result (int32): " << result_sub_int32 << std::endl;
    std::cout << "Multiplication result (int32): " << result_mul_int32 << std::endl;

    // More test cases with other types
    auto result_add_uint32 = *tensor_uint32 + variant_uint32;
    auto result_sub_uint32 = *tensor_uint32 - variant_uint32;
    auto result_mul_uint32 = *tensor_uint32 * variant_uint32;

    auto result_add_int8 = *tensor_int8 + variant_int8;
    auto result_sub_int8 = *tensor_int8 - variant_int8;
    auto result_mul_int8 = *tensor_int8 * variant_int8;

    // Print more results
    std::cout << "Addition result (uint32): " << result_add_uint32 << std::endl;
    std::cout << "Subtraction result (uint32): " << result_sub_uint32 << std::endl;
    std::cout << "Multiplication result (uint32): " << result_mul_uint32 << std::endl;

    std::cout << "Addition result (int8): " << result_add_int8 << std::endl;
    std::cout << "Subtraction result (int8): " << result_sub_int8 << std::endl;
    std::cout << "Multiplication result (int8): " << result_mul_int8 << std::endl;
}

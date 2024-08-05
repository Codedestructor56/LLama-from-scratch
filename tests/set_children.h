#pragma once

#include "tensor.h" 
#include <iostream>
#include <memory>
#include <vector>


void test_set_children() {
    // Create some sample child tensors with different data types
    auto child1 = std::make_shared<Tensor<FLOAT32>>(std::vector<int>{2, 2});
    auto child2 = std::make_shared<Tensor<UINT8>>(std::vector<int>{2, 2});
    auto child3 = std::make_shared<Tensor<INT32>>(std::vector<int>{2, 2});

    // Create a parent tensor
    Tensor<FLOAT32> parent_tensor(std::vector<int>{2, 2});

    // Create a vector of TensorVariant to hold the child tensors
    std::vector<TensorVariant> children = {
        child1,
        child2,
        child3
    };

    // Set children using the set_children method
    parent_tensor.set_children(children);

    // Print the number of children to verify
    std::cout << "Number of children in parent tensor: " << parent_tensor.get_children_size() << std::endl;

    // Verify the children are set correctly
    for (const auto& child : parent_tensor.get_children()) {
        std::visit([](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<Tensor<FLOAT32>>>) {
                std::cout << "Child tensor type: FLOAT32" << std::endl; 
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Tensor<UINT8>>>) {
                std::cout << "Child tensor type: UINT8" << std::endl; 
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Tensor<INT32>>>) {
                std::cout << "Child tensor type: INT32" << std::endl; 
            } 
            // Add more type checks as needed
        }, child);
    }
}

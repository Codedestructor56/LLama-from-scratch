#include <iostream>
#include <vector>
#include <cassert>
#include "tensor.h"

void test_reshape() {
    // Test 1: Successful reshape
    Tensor<UINT8> tensor1({1, 2, 3, 4, 5, 6}, {2, 3});
    tensor1.reshape({3, 2});
    assert(tensor1.shape == std::vector<int>({3, 2}));
    std::cout << "Test 1 passed: reshape({3, 2})\n";

    // Test 2: Reshape to original shape
    tensor1.reshape({2, 3});
    assert(tensor1.shape == std::vector<int>({2, 3}));
    std::cout << "Test 2 passed: reshape({2, 3})\n";

    // Test 3: Invalid reshape - different number of elements
    bool caught_exception = false;
    try {
        tensor1.reshape({2, 4});
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        assert(std::string(e.what()) == "The new shape does not match the tensor's shape");
    }
    assert(caught_exception);
    std::cout << "Test 3 passed: reshape({2, 4}) throws exception\n";

    // Test 4: Successful reshape with -1
    Tensor<UINT8> tensor2({1, 2, 3, 4, 5, 6}, {2, 3});
    tensor2.reshape({-1, 2});
    assert(tensor2.shape == std::vector<int>({3, 2}));
    std::cout << "Test 4 passed: reshape({-1, 2})\n";

    // Test 5: Another successful reshape with -1
    tensor2.reshape({2, -1});
    assert(tensor2.shape == std::vector<int>({2, 3}));
    std::cout << "Test 5 passed: reshape({2, -1})\n";

    // Test 6: Invalid reshape with multiple -1
    caught_exception = false;
    try {
        tensor2.reshape({-1, -1});
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        assert(std::string(e.what()) == "Only one dimension can be inferred");
    }
    assert(caught_exception);
    std::cout << "Test 6 passed: reshape({-1, -1}) throws exception\n";

    // Test 7: Reshape to a single dimension
    tensor1.reshape({6});
    assert(tensor1.shape == std::vector<int>({6}));
    std::cout << "Test 7 passed: reshape({6})\n";

    // Test 8: Reshape with inferred dimension
    tensor1.reshape({1, -1});
    assert(tensor1.shape == std::vector<int>({1, 6}));
    std::cout << "Test 8 passed: reshape({1, -1})\n";

    std::cout << "All tests passed!\n";
}


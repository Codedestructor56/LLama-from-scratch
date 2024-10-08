#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>
#include "tensor.h"

void benchmarkTensorOperations(int num_elems) {
    using T = float;
    const T tolerance = 1e-5;

    // Create random data
    std::vector<int> shape = {num_elems};
    std::vector<T> vec1(num_elems);
    std::vector<T> vec2(num_elems);

    std::mt19937 rng(12345); // Seed for reproducibility
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < num_elems; ++i) {
        vec1[i] = static_cast<T>(dist(rng));
        vec2[i] = static_cast<T>(dist(rng));
    }

    // Create tensors
    Tensor<FLOAT32> tensor1(vec1, shape);
    Tensor<FLOAT32> tensor2(vec2, shape);

    // Test CPU performance
    tensor1.change_device(CPU);
    tensor2.change_device(CPU);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor<FLOAT32> cpu_result_add = tensor1 + tensor2;
    Tensor<FLOAT32> cpu_result_sub = tensor1 - tensor2;
    Tensor<FLOAT32> cpu_result_mul = tensor1 * tensor2;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // Test CUDA performance
    tensor1.change_device(CUDA);
    tensor2.change_device(CUDA);

    auto start_cuda = std::chrono::high_resolution_clock::now();
    Tensor<FLOAT32> cuda_result_add = tensor1 + tensor2;
    Tensor<FLOAT32> cuda_result_sub = tensor1 - tensor2;
    Tensor<FLOAT32> cuda_result_mul = tensor1 * tensor2;
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;

    std::cout << "CUDA time: " << cuda_duration.count() << " seconds" << std::endl;

    // Assertions to compare CPU and CUDA results
    for (int i = 0; i < num_elems; ++i) {
        assert(std::abs(cpu_result_add.data()[i] - cuda_result_add.data()[i]) < tolerance);
        assert(std::abs(cpu_result_sub.data()[i] - cuda_result_sub.data()[i]) < tolerance);
        assert(std::abs(cpu_result_mul.data()[i] - cuda_result_mul.data()[i]) < tolerance);
    }

    std::cout << "CPU and CUDA results match within tolerance." << std::endl;

    // Performance improvement
    double speedup = cpu_duration.count() / cuda_duration.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
}

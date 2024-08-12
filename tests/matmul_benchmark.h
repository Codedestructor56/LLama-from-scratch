#include <cassert>
#include <chrono>
#include <iostream>

void benchmark_matmul() {
    // Define the shape of the matrices
    std::vector<int> shape1 = {6, 6};
    std::vector<int> shape2 = {6, 6};

    // Initialize the matrices with random values
    Tensor<FLOAT32> tensor1 = Tensor<FLOAT32>::rand(shape1);
    Tensor<FLOAT32> tensor2 = Tensor<FLOAT32>::rand(shape2);

    // Set the tensors to CPU and perform matmul on CPU
    tensor1.change_device(CPU);
    tensor2.change_device(CPU);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor<FLOAT32> result_cpu = matmul(tensor1, tensor2);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Set the tensors to CUDA and perform matmul on CUDA
    tensor1.change_device(CUDA);
    tensor2.change_device(CUDA);
    auto start_cuda = std::chrono::high_resolution_clock::now();
    Tensor<FLOAT32> result_cuda = matmul(tensor1, tensor2);
    auto end_cuda = std::chrono::high_resolution_clock::now();

    // Convert the CUDA result back to CPU to compare with the CPU result
    result_cuda.change_device(CPU);
    std::cout<<"CPU result: "<<result_cpu<<std::endl;
    std::cout<<"CUDA result: "<<result_cuda<<std::endl;
    // Check if the results are the same using assertions
    int num_elements = std::accumulate(shape1.begin(), shape1.end(), 1, std::multiplies<int>());
    for (int i = 0; i < num_elements; ++i) {
        assert(std::abs(result_cpu.data()[i] - result_cuda.data()[i]) < 1e-5);
    }

    // Output the timings
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();

    std::cout << "CPU MatMul Time: " << duration_cpu << " ms" << std::endl;
    std::cout << "CUDA MatMul Time: " << duration_cuda << " ms" << std::endl;
}

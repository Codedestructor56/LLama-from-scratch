#include <cuda_runtime.h>
#include "tensor.h"

__global__ void atomicMulKernel(float* data, float* values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        //atomicmul(&data[idx], values[idx]);
        data[idx] *= values[idx];
    }
}

void atomicMulTensor(Tensor<FLOAT32>& tensor, const Tensor<FLOAT32>& values) {
    if (tensor.shape != values.shape) {
        throw std::runtime_error("Shapes of the tensors do not match!");
    }

    int numElements = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<int>());

    float* tensorData = tensor.data();
    float* valuesData = values.data();
    
    std::cout<<"Within the kernel: "<<tensor<<std::endl;
    std::cout<<"Within the kernel, value: "<<values<<std::endl;
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    atomicMulKernel<<<numBlocks, blockSize>>>(tensorData, valuesData, numElements);
cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
} 
  cudaDeviceSynchronize();
  std::cout<<"result: "<<std::endl;
    for(int i = 0; i<numElements; i++){
      std::cout<<tensorData[i]<<std::endl;
    }

}



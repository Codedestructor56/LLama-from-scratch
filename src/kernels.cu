#include <cuda_runtime.h>
#include "tensor.h"

template <typename Op>
struct IsMulOperation {
    static const bool value = false;
};

template <>
struct IsMulOperation<std::multiplies<float>> {
    static const bool value = true;
};

template <typename Op>
struct IsAddOrSubOperation {
    static const bool value = false;
};

template <>
struct IsAddOrSubOperation<std::plus<float>> {
    static const bool value = true;
};

template <>
struct IsAddOrSubOperation<std::minus<float>> {
    static const bool value = true;
};

__device__ void atomicmul(float* address, float val) {
    float old = *address;
    float assumed;
    do {
        assumed = old;
        old = atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(assumed * val));
    } while (__float_as_int(*address) != __float_as_int(assumed));
}

__global__ void atomicMulKernel(float* data, float* values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicmul(&data[idx], values[idx]);
    }
}



void atomicMulTensor(Tensor<FLOAT32>& tensor, const Tensor<FLOAT32>& values) {
    if (tensor.shape != values.shape) {
        throw std::runtime_error("Shapes of the tensors do not match!");
    }

    int numElements = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<int>());

    float* d_tensorData;
    float* d_valuesData;
    cudaMalloc(&d_tensorData, numElements * sizeof(float));
    cudaMalloc(&d_valuesData, numElements * sizeof(float));

    cudaMemcpy(d_tensorData, tensor.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valuesData, values.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    atomicMulKernel<<<numBlocks, blockSize>>>(d_tensorData, d_valuesData, numElements);
    cudaMemcpy(tensor.data(), d_tensorData, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_tensorData);
    cudaFree(d_valuesData);

    cudaDeviceSynchronize();
}

template <typename T>
__device__ void AtomicMul(T* address, T val) {
    if constexpr (std::is_same_v<T, float>) {
        atomicmul(address, val);
    } else {
        atomicmul(reinterpret_cast<float*>(address), static_cast<float>(val));
    }
}

template<typename T, typename Op>
__global__ void tensorOperationKernel(const T* a, const T* b, T* res, int num_elems, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        if (IsMulOperation<Op>::value) {
            AtomicMul(&res[idx], op(a[idx], b[idx]));
        } else if (IsAddOrSubOperation<Op>::value) {
            atomicAdd(reinterpret_cast<int32_t*>(&res[idx]), static_cast<int32_t>(op(a[idx], b[idx])));
        } else {
            res[idx] = op(a[idx], b[idx]);
        }
    }
}


template <typename T, typename Op>
void tensorOperationCuda(const T* h_a, const T* h_b, T* h_result, int num_elems, Op op, int block_size) {
    int grid_size = (num_elems + block_size - 1) / block_size;

    T* d_a;
    T* d_b;
    T* d_result;

    cudaMalloc(&d_a, num_elems * sizeof(T));
    cudaMalloc(&d_b, num_elems * sizeof(T));
    cudaMalloc(&d_result, num_elems * sizeof(T));

    cudaMemcpy(d_a, h_a, num_elems * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_elems * sizeof(T), cudaMemcpyHostToDevice);

    tensorOperationKernel<<<grid_size, block_size>>>(d_a, d_b, d_result, num_elems, op);
    cudaDeviceSynchronize(); 

    cudaMemcpy(h_result, d_result, num_elems * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int n, int p) {
    extern __shared__ float sharedMem[];

    float* Asub = sharedMem;
    float* Bsub = sharedMem + blockDim.y * blockDim.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        if (row < m && tile * blockDim.x + threadIdx.x < n) {
            Asub[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + tile * blockDim.x + threadIdx.x];
        } else {
            Asub[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (col < p && tile * blockDim.y + threadIdx.y < n) {
            Bsub[threadIdx.y * blockDim.x + threadIdx.x] = B[(tile * blockDim.y + threadIdx.y) * p + col];
        } else {
            Bsub[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += Asub[threadIdx.y * blockDim.x + k] * Bsub[k * blockDim.x + threadIdx.x]; 
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}


template <typename T>
void matmul_cuda(const T* h_A, const T* h_B, T* h_C, int m, int n, int p) {
    T* d_A;
    T* d_B;
    T* d_C;

    cudaMalloc(&d_A, m * n * sizeof(T));
    cudaMalloc(&d_B, n * p * sizeof(T));
    cudaMalloc(&d_C, m * p * sizeof(T));

    cudaMemcpy(d_A, h_A, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * p * sizeof(T), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); 
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y); 
    size_t sharedMemSize = 2 * blockDim.x * blockDim.y * sizeof(T);

    matmul_kernel<<<gridDim, blockDim, sharedMemSize>>>(reinterpret_cast<const float*>(d_A), 
                                                       reinterpret_cast<const float*>(d_B),
                                                       reinterpret_cast<float*>(d_C), m, n, p);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, m * p * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template void matmul_cuda<float>(const float*, const float*, float*, int, int, int);
template void matmul_cuda<int32_t>(const int32_t*, const int32_t*, int32_t*, int, int, int);
template void matmul_cuda<uint16_t>(const uint16_t*, const uint16_t*, uint16_t*, int, int, int);
template void matmul_cuda<uint8_t>(const uint8_t*, const uint8_t*, uint8_t*, int, int, int);
template void matmul_cuda<uint32_t>(const uint32_t*, const uint32_t*, uint32_t*, int, int, int);
template void matmul_cuda<int8_t>(const int8_t*, const int8_t*, int8_t*, int, int, int);


template void tensorOperationCuda<float, std::plus<float>>(const float*, const float*, float*, int, std::plus<float>, int);
template void tensorOperationCuda<float, std::minus<float>>(const float*, const float*, float*, int, std::minus<float>, int);
template void tensorOperationCuda<float, std::multiplies<float>>(const float*, const float*, float*, int, std::multiplies<float>, int);

template void tensorOperationCuda<int32_t, std::plus<int32_t>>(const int32_t*, const int32_t*, int32_t*, int, std::plus<int32_t>, int);
template void tensorOperationCuda<int32_t, std::minus<int32_t>>(const int32_t*, const int32_t*, int32_t*, int, std::minus<int32_t>, int);
template void tensorOperationCuda<int32_t, std::multiplies<int32_t>>(const int32_t*, const int32_t*, int32_t*, int, std::multiplies<int32_t>, int);

template void tensorOperationCuda<uint16_t, std::plus<uint16_t>>(const uint16_t*, const uint16_t*, uint16_t*, int, std::plus<uint16_t>, int);
template void tensorOperationCuda<uint16_t, std::minus<uint16_t>>(const uint16_t*, const uint16_t*, uint16_t*, int, std::minus<uint16_t>, int);
template void tensorOperationCuda<uint16_t, std::multiplies<uint16_t>>(const uint16_t*, const uint16_t*, uint16_t*, int, std::multiplies<uint16_t>, int);

template void tensorOperationCuda<uint8_t, std::plus<uint8_t>>(const uint8_t*, const uint8_t*, uint8_t*, int, std::plus<uint8_t>, int);
template void tensorOperationCuda<uint8_t, std::minus<uint8_t>>(const uint8_t*, const uint8_t*, uint8_t*, int, std::minus<uint8_t>, int);
template void tensorOperationCuda<uint8_t, std::multiplies<uint8_t>>(const uint8_t*, const uint8_t*, uint8_t*, int, std::multiplies<uint8_t>, int);

template void tensorOperationCuda<uint32_t, std::plus<uint32_t>>(const uint32_t*, const uint32_t*, uint32_t*, int, std::plus<uint32_t>, int);
template void tensorOperationCuda<uint32_t, std::minus<uint32_t>>(const uint32_t*, const uint32_t*, uint32_t*, int, std::minus<uint32_t>, int);
template void tensorOperationCuda<uint32_t, std::multiplies<uint32_t>>(const uint32_t*, const uint32_t*, uint32_t*, int, std::multiplies<uint32_t>, int);

template void tensorOperationCuda<int8_t, std::plus<int8_t>>(const int8_t*, const int8_t*, int8_t*, int, std::plus<int8_t>, int);
template void tensorOperationCuda<int8_t, std::minus<int8_t>>(const int8_t*, const int8_t*, int8_t*, int, std::minus<int8_t>, int);
template void tensorOperationCuda<int8_t, std::multiplies<int8_t>>(const int8_t*, const int8_t*, int8_t*, int, std::multiplies<int8_t>, int);

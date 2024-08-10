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

template <typename T>
__device__ void atomicMul(T* address, T val) {
    if constexpr (std::is_same_v<T, float>) {
        atomicMul(address, val);
    } else {
        *address *= val;
    }
}

template<typename T, typename Op>
__global__ void tensorOperationKernel(const T* a, const T* b, T* res, int num_elems, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) {
        if (IsMulOperation<Op>::value) {
            atomicMul(&res[idx], op(a[idx], b[idx]));
        } else if (IsAddOrSubOperation<Op>::value) {
            atomicAdd(reinterpret_cast<int32_t*>(&res[idx]), static_cast<int32_t>(op(a[idx], b[idx])));
        } else {
            res[idx] = op(a[idx], b[idx]);
        }
    }
}

template <typename T, typename Op>
void tensorOperationCuda(const T* a, const T* b, T* result, int num_elems, Op op, int block_size) {
    int grid_size = (num_elems + block_size - 1) / block_size;
    tensorOperationKernel<<<grid_size, block_size>>>(a, b, result, num_elems, op);
    cudaDeviceSynchronize(); 
}

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

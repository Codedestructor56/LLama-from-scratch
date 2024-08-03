#include "tensor.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <typeinfo>
#include <cxxabi.h>

size_t get_dtype_size(DType dtype) {
    switch(dtype) {
        case FLOAT16: return 2; // Half-precision float, usually 2 bytes
        case FLOAT32: return 4; // Single-precision float, 4 bytes
        case INT8:    return 1; // 8-bit signed integer, 1 byte
        case INT32:   return 4; // 32-bit signed integer, 4 bytes
        case UINT8:   return 1; // 8-bit unsigned integer, 1 byte
        case UINT32:  return 4; // 32-bit unsigned integer, 4 bytes
        default:      return 0; // Unknown type
    }
}

template<typename T>
std::string type_name() {
    int status;
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
        std::free
    };
    return (status == 0) ? res.get() : typeid(T).name();
}

void* allocate_memory(DType dtype, size_t num_elements) {
    size_t size = get_dtype_size(dtype);
    if (size == 0) {
        return NULL; 
    }
    return malloc(size * num_elements);
}

void deallocate_memory(void* ptr) {
    free(ptr);
}

template<DType dtype>
void Tensor<dtype>::allocate_and_initialize(const std::vector<int>& shape, bool zero_initialize, bool is_rand) {
    int num_elements = 1;
    for (auto elem : shape) {
        num_elements *= elem;
    }
    this->shape = shape;
    T* arr = static_cast<T*>(allocate_memory(dtype, num_elements));
    if (zero_initialize) {
        std::memset(arr, 0, num_elements * sizeof(T));
        for (int i = 0; i < std::min(num_elements, 10); ++i) {
            std::cout << "Zero init arr[" << i << "] = " << +arr[i] << std::endl;
        }
    } else {
        if (is_rand) {
            if (dtype != FLOAT32) {
                throw std::runtime_error("Random initialization is only supported for FLOAT32 dtype.");
            }
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            for (int i = 0; i < num_elements; ++i) {
                arr[i] = static_cast<T>(dis(gen)); 
            }
        } else {
            std::fill_n(arr, num_elements, T(1));
            for (int i = 0; i < std::min(num_elements, 10); ++i) {
                std::cout << "One init arr[" << i << "] = " << +arr[i] << std::endl;
            }
        }
    }
    this->data_ = arr;
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::ones(const std::vector<int>& shape) {
    Tensor<dtype> tensor;
    tensor.allocate_and_initialize(shape, false, false);
    return tensor;
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::zeros(const std::vector<int>& shape) {
    Tensor tensor;
    tensor.allocate_and_initialize(shape, true, false);
    return tensor;
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::rand(const std::vector<int>& shape) {
    Tensor tensor;
    tensor.allocate_and_initialize(shape, false, true);
    return tensor;
}


template<DType dtype>
typename Tensor<dtype>::T Tensor<dtype>::get(const std::vector<int>& indices) const {
    if (indices.size() != this->shape.size()) {
        throw std::runtime_error("Shapes do not match for simple get op");
    }
    int flat_index = 0;
    int multiplier = 1;
    for (int i = this->shape.size() - 1; i >= 0; --i) {
        if(indices[i]>this->shape[i]-1){
          throw std::runtime_error("Index out of range");
        }
        flat_index += indices[i] * multiplier;
        multiplier *= this->shape[i];
    }
    return this->data_[flat_index];
}

template<DType dtype>
void Tensor<dtype>::set(const std::vector<int>& indices, const typename Tensor<dtype>::T& value){
    if (indices.size() != this->shape.size()) {
        throw std::runtime_error("Shapes do not match for simple set op");
    }
    using C = typename DTypeToType<dtype>::Type; 
    //std::cout<<"The set value: "<<std::is_same<T,C>::value<<std::endl;
    //std::cout << "Type of T: " << type_name<T>() << std::endl;
    //std::cout << "Type of C: " << type_name<C>() << std::endl;
    if(!std::is_same<T,C>::value){
       throw std::runtime_error("Incompatible type for the value you just set");
    }
    int flat_index = 0;
    int multiplier = 1;
    for (int i = this->shape.size() - 1; i >= 0; --i) {
        if(indices[i]>this->shape[i]-1){
          throw std::runtime_error("Index out of range");
        }
        flat_index += indices[i] * multiplier;
        multiplier *= this->shape[i];
    }
    this->data_[flat_index]=value;
}

template<DType dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    os << "Tensor of type " << tensor.type << " with shape [";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        os << tensor.shape[i];
        if (i < tensor.shape.size() - 1) {
            os << ", ";
        }
    }
    os << "] and data: [";
    int product = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<int>());
    for (size_t i = 0; i < product; ++i) {
        if (tensor.type == DType::UINT8 || tensor.type == DType::INT8){ 
          os << static_cast<int>(tensor.data()[i]);
        }
        else{
          os <<tensor.data()[i];
        }
        if (i < product - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Explicit template instantiation
template class Tensor<FLOAT16>;
template class Tensor<FLOAT32>;
template class Tensor<INT8>;
template class Tensor<INT32>;
template class Tensor<UINT8>;
template class Tensor<UINT32>;

template std::ostream& operator<< <FLOAT16>(std::ostream& os, const Tensor<FLOAT16>& tensor);
template std::ostream& operator<< <FLOAT32>(std::ostream& os, const Tensor<FLOAT32>& tensor);
template std::ostream& operator<< <INT8>(std::ostream& os, const Tensor<INT8>& tensor);
template std::ostream& operator<< <INT32>(std::ostream& os, const Tensor<INT32>& tensor);
template std::ostream& operator<< <UINT8>(std::ostream& os, const Tensor<UINT8>& tensor);
template std::ostream& operator<< <UINT32>(std::ostream& os, const Tensor<UINT32>& tensor);


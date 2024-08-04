#include "tensor.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <typeinfo>
#include <cxxabi.h>
#include <iomanip> 

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
Tensor<dtype> Tensor<dtype>::get_slice(const std::vector<int>& start_indices, const std::vector<int>& end_indices, const std::vector<int>& stride) const{
    if (start_indices.size() != shape.size() || end_indices.size() != shape.size()) {
        throw std::runtime_error("start_indices and end_indices must have the same size as the tensor's shape");
    }
    std::vector<int> result_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        int end = (end_indices[i] == -1) ? shape[i] : end_indices[i];
        result_shape[i] = (end - start_indices[i]) / (stride.size() > i ? stride[i] : 1);
        if(end>shape[i]){
          throw std::runtime_error("Index out of bounds");
        }
        if (result_shape[i] < 0) {
            throw std::runtime_error("Invalid slice indices or stride for dimension " + std::to_string(i));
        }
    }
    size_t num_elements = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<int>());
    std::vector<T> result_data(num_elements);
    auto get_flat_index = [&](const std::vector<int>& indices) -> size_t {
        size_t flat_index = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        return flat_index;
    };
    std::vector<int> current_indices(shape.size(), 0);
    std::vector<int> result_indices(shape.size(), 0);

    auto increment_indices = [&](std::vector<int>& indices, const std::vector<int>& shape, const std::vector<int>& stride) -> bool {
        for (int i = shape.size() - 1; i >= 0; --i) {
            indices[i] += (stride.size() > i ? stride[i] : 1);
            if (indices[i] < shape[i]) {
                return true;
            }
            indices[i] = 0;
        }
        return false;
    };
    for (size_t i = 0; i < num_elements; ++i) {
        for (size_t j = 0; j < shape.size(); ++j) {
            current_indices[j] = start_indices[j] + result_indices[j] * (stride.size() > j ? stride[j] : 1);
        }
        result_data[i] = data_[get_flat_index(current_indices)];
        if (!increment_indices(result_indices, result_shape, std::vector<int>(shape.size(), 1))) {
            break;
        }
    }

    return Tensor<dtype>(result_data, result_shape);
}

template<DType dtype>
void Tensor<dtype>::set_slice(const std::vector<int>& start_indices, const std::vector<int>& end_indices, const std::vector<T>& values) { 
    if (start_indices.size() != end_indices.size() || start_indices.size() != this->shape.size()) {
        throw std::invalid_argument("Dimension mismatch between start indices, end indices, and tensor shape.");
    }

    int slice_size = 1;
    for (size_t i = 0; i < start_indices.size(); ++i) {
        int end = (end_indices[i] == -1) ? shape[i] : end_indices[i];
        int slice_dim_size = end - start_indices[i];

        if (slice_dim_size <= 0) {
            throw std::invalid_argument("End indices must be greater than start indices.");
        }
        slice_size *= slice_dim_size;
    }

    if (slice_size != values.size()) {
        throw std::runtime_error("Number of elements in the values vector does not match the number of elements in the slice.");
    }

    auto compute_linear_index = [this](const std::vector<int>& indices) -> int {
        int linear_index = 0;
        int stride = 1;
        for (int i = this->shape.size() - 1; i >= 0; --i) {
            linear_index += indices[i] * stride;
            stride *= this->shape[i];
        }
        return linear_index;
    };
 
    std::vector<int> current_indices = start_indices;
    for (size_t i = 0; i < values.size(); ++i) {
        int linear_index = compute_linear_index(current_indices);
        this->data_[linear_index] = values[i];
 
        for (int j = current_indices.size() - 1; j >= 0; --j) {
            current_indices[j]++;
            if (current_indices[j] < end_indices[j]) {
                break;
            }
            current_indices[j] = start_indices[j];
        }
    }
}

template<DType dtype>
void print_tensor_data(std::ostream& os, const std::vector<int>& shape, const std::vector<int>& indices, const typename DTypeToType<dtype>::Type* data, int depth) {
    if (depth == shape.size() - 1) {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            std::vector<int> new_indices = indices;
            new_indices.push_back(i);
            int flat_index = 0;
            int multiplier = 1;
            for (int j = shape.size() - 1; j >= 0; --j) {
                flat_index += new_indices[j] * multiplier;
                multiplier *= shape[j];
            }
            if (dtype == DType::UINT8 || dtype == DType::INT8) {
                os << static_cast<int>(data[flat_index]);
            } else {
                os << data[flat_index];
            }
            if (i < shape[depth] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            std::vector<int> new_indices = indices;
            new_indices.push_back(i);
            print_tensor_data<dtype>(os, shape, new_indices, data, depth + 1);
            if (i < shape[depth] - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
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
    os << "] and data: ";
    print_tensor_data<dtype>(os, tensor.shape, {}, tensor.data(), 0);
    return os;
}

template std::ostream& operator<<(std::ostream& os, const Tensor<FLOAT16>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<FLOAT32>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<INT8>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<INT32>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<UINT8>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<UINT32>& tensor);


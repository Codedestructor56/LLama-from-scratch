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

std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case FLOAT16: return "FLOAT16";
        case FLOAT32: return "FLOAT32";
        case INT8: return "INT8";
        case INT32: return "INT32";
        case UINT8: return "UINT8";
        case UINT32: return "UINT32";
        default: return "Unknown DType";
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
        int end = (end_indices[i] == -1) ? this->shape[i] : end_indices[i];
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
template <typename Op>
Tensor<dtype> Tensor<dtype>::tensorOperation(const TensorVariant& rhs, Op op) const {
    if (auto other_tensor = std::get_if<std::shared_ptr<Tensor<dtype>>>(&rhs)) {
        if (this->shape != (*other_tensor)->shape) {
            throw std::runtime_error("Shapes do not match for tensor operation.");
        }

        Tensor<dtype> result(this->shape);
        int num_elems = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());

        for (int i = 0; i < num_elems; ++i) {
            result.data()[i] = op(this->data()[i], (*other_tensor)->data()[i]);
        }
        
        result.type = dtype;
        result.set_children(this->get_children());
        return result;
    }
    throw std::runtime_error("DType mismatch for tensor operation.");
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const {
    return tensorOperation(std::make_shared<Tensor<dtype>>(other), std::plus<typename DTypeToType<dtype>::Type>());
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const {
    return tensorOperation(std::make_shared<Tensor<dtype>>(other), std::minus<typename DTypeToType<dtype>::Type>());
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const {
    return tensorOperation(std::make_shared<Tensor<dtype>>(other), std::multiplies<typename DTypeToType<dtype>::Type>());
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator+(const TensorVariant& other) const {
    return tensorOperation(other, std::plus<typename DTypeToType<dtype>::Type>());
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator-(const TensorVariant& other) const {
    return tensorOperation(other, std::minus<typename DTypeToType<dtype>::Type>());
}

template <DType dtype>
Tensor<dtype> Tensor<dtype>::operator*(const TensorVariant& other) const {
    return tensorOperation(other, std::multiplies<typename DTypeToType<dtype>::Type>());
}


template<DType dtype>
void Tensor<dtype>::reshape(const std::vector<int>& new_shape) {
    std::vector<int> mutable_new_shape = new_shape;
    int orig_elems = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());
    int new_elems = 1;
    int infer_index = -1;
 
    for (int i = 0; i < mutable_new_shape.size(); i++) {
        if (mutable_new_shape[i] == -1) {
            if (infer_index != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_index = i;
        } else {
            new_elems *= mutable_new_shape[i];
        }
    }

    if (infer_index != -1) {
        if (orig_elems % new_elems != 0) {
            throw std::runtime_error("The new shape does not match the tensor's shape");
        }
        mutable_new_shape[infer_index] = orig_elems / new_elems;
    } else if (new_elems != orig_elems) {
        throw std::runtime_error("The new shape does not match the tensor's shape");
    }

    this->shape = mutable_new_shape;
}

template<DType dtype>
Tensor<dtype> matmul(const Tensor<dtype>& tens1, const Tensor<dtype>& tens2) {
    using T = typename DTypeToType<dtype>::Type;

    if (tens1.shape.size() < 2 || tens2.shape.size() < 2) {
        throw std::runtime_error("matmul requires both tensors to be at least 2-dimensional");
    }

    int axis1 = tens1.shape.size() - 1;
    int axis2 = tens2.shape.size() - 2;

    if (tens1.shape[axis1] != tens2.shape[axis2]) {
        throw std::runtime_error("Inner dimensions must match for matrix multiplication");
    }

    std::vector<int> result_shape;
    result_shape.insert(result_shape.end(), tens1.shape.begin(), tens1.shape.end() - 1);
    result_shape.insert(result_shape.end(), tens2.shape.begin(), tens2.shape.end() - 2);
    result_shape.push_back(tens2.shape.back());

    Tensor<dtype> result(result_shape);
    T* result_data = result.data();

    const T* data1 = tens1.data();
    const T* data2 = tens2.data();

    int m = std::accumulate(tens1.shape.begin(), tens1.shape.end() - 1, 1, std::multiplies<int>());
    int n = tens1.shape.back();
    int p = tens2.shape.back();

    std::fill(result_data, result_data + result_shape.back() * m, T(0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                int result_index = i * p + j;
                int data1_index = i * n + k;
                int data2_index = k * p + j;
                result_data[result_index] += data1[data1_index] * data2[data2_index];
            }
        }
    }
    result.type = dtype;
    return result;
}

template <DType dtype>
Tensor<dtype> hstack_impl(const std::vector<Tensor<dtype>>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("No tensors provided for hstack.");
    }

    int rows = tensors[0].shape[0];
    int total_cols = 0;
    for (const auto& tensor : tensors) {
        if (tensor.shape.size() != 2 || tensor.shape[0] != rows) {
            throw std::runtime_error("Incompatible tensor dimensions for hstack.");
        }
        total_cols += tensor.shape[1];
    }

    std::vector<int> new_shape = {rows, total_cols};
    Tensor<dtype> result(new_shape);

    int col_offset = 0;
    for (const auto& tensor : tensors) {
        for (int i = 0; i < rows; ++i) {
            std::copy(tensor.data() + i * tensor.shape[1], tensor.data() + (i + 1) * tensor.shape[1], result.data() + i * total_cols + col_offset);
        }
        col_offset += tensor.shape[1];
    }

    return result;
}

template <DType dtype>
Tensor<dtype> hstack(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2) {
    return hstack_impl<dtype>({tensor1, tensor2});
}

template <DType dtype>
Tensor<dtype> hstack(const std::vector<Tensor<dtype>>& tensors) {
    return hstack_impl<dtype>(tensors);
}

template <DType dtype>
Tensor<dtype> vstack_impl(const std::vector<Tensor<dtype>>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("No tensors provided for vstack.");
    }

    int cols = tensors[0].shape[1];
    int total_rows = 0;
    for (const auto& tensor : tensors) {
        if (tensor.shape.size() != 2 || tensor.shape[1] != cols) {
            throw std::runtime_error("Incompatible tensor dimensions for vstack.");
        }
        total_rows += tensor.shape[0];
    }

    std::vector<int> new_shape = {total_rows, cols};
    Tensor<dtype> result(new_shape);

    int row_offset = 0;
    for (const auto& tensor : tensors) {
        std::copy(tensor.data(), tensor.data() + tensor.shape[0] * cols, result.data() + row_offset * cols);
        row_offset += tensor.shape[0];
    }

    return result;
}

template <DType dtype>
Tensor<dtype> vstack(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2) {
    return vstack_impl<dtype>({tensor1, tensor2});
}

template <DType dtype>
Tensor<dtype> vstack(const std::vector<Tensor<dtype>>& tensors) {
    return vstack_impl<dtype>(tensors);
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
    os << "Tensor of type " << dtype_to_string(tensor.type)  << " with shape [";
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

template Tensor<FLOAT16> hstack(const std::vector<Tensor<FLOAT16>>& tensors);
template Tensor<FLOAT32> hstack(const std::vector<Tensor<FLOAT32>>& tensors);
template Tensor<INT8> hstack(const std::vector<Tensor<INT8>>& tensors);
template Tensor<INT32> hstack(const std::vector<Tensor<INT32>>& tensors);
template Tensor<UINT8> hstack(const std::vector<Tensor<UINT8>>& tensors);
template Tensor<UINT32> hstack(const std::vector<Tensor<UINT32>>& tensors);

template Tensor<FLOAT16> hstack(const Tensor<FLOAT16>& tensor1, const Tensor<FLOAT16>& tensor2);
template Tensor<FLOAT32> hstack(const Tensor<FLOAT32>& tensor1, const Tensor<FLOAT32>& tensor2); 
template Tensor<INT8> hstack(const Tensor<INT8>& tensor1, const Tensor<INT8>& tensor2); 
template Tensor<INT32> hstack(const Tensor<INT32>& tensor1, const Tensor<INT32>& tensor2); 
template Tensor<UINT8> hstack(const Tensor<UINT8>& tensor1, const Tensor<UINT8>& tensor2); 
template Tensor<UINT32> hstack(const Tensor<UINT32>& tensor1, const Tensor<UINT32>& tensor2); 

template Tensor<FLOAT16> vstack(const std::vector<Tensor<FLOAT16>>& tensors);
template Tensor<FLOAT32> vstack(const std::vector<Tensor<FLOAT32>>& tensors);
template Tensor<INT8> vstack(const std::vector<Tensor<INT8>>& tensors);
template Tensor<INT32> vstack(const std::vector<Tensor<INT32>>& tensors);
template Tensor<UINT8> vstack(const std::vector<Tensor<UINT8>>& tensors);
template Tensor<UINT32> vstack(const std::vector<Tensor<UINT32>>& tensors);

template Tensor<FLOAT16> vstack(const Tensor<FLOAT16>& tensor1, const Tensor<FLOAT16>& tensor2);
template Tensor<FLOAT32> vstack(const Tensor<FLOAT32>& tensor1, const Tensor<FLOAT32>& tensor2); 
template Tensor<INT8> vstack(const Tensor<INT8>& tensor1, const Tensor<INT8>& tensor2); 
template Tensor<INT32> vstack(const Tensor<INT32>& tensor1, const Tensor<INT32>& tensor2); 
template Tensor<UINT8> vstack(const Tensor<UINT8>& tensor1, const Tensor<UINT8>& tensor2); 
template Tensor<UINT32> vstack(const Tensor<UINT32>& tensor1, const Tensor<UINT32>& tensor2); 

template Tensor<FLOAT16> matmul<FLOAT16>(const Tensor<FLOAT16>&, const Tensor<FLOAT16>&);
template Tensor<FLOAT32> matmul<FLOAT32>(const Tensor<FLOAT32>&, const Tensor<FLOAT32>&);
template Tensor<INT8> matmul<INT8>(const Tensor<INT8>&, const Tensor<INT8>&);
template Tensor<INT32> matmul<INT32>(const Tensor<INT32>&, const Tensor<INT32>&);
template Tensor<UINT8> matmul<UINT8>(const Tensor<UINT8>&, const Tensor<UINT8>&);
template Tensor<UINT32> matmul<UINT32>(const Tensor<UINT32>&, const Tensor<UINT32>&);

template std::ostream& operator<<(std::ostream& os, const Tensor<FLOAT16>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<FLOAT32>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<INT8>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<INT32>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<UINT8>& tensor);
template std::ostream& operator<<(std::ostream& os, const Tensor<UINT32>& tensor);


template Tensor<FLOAT16> Tensor<FLOAT16>::operator+(const TensorVariant& other) const;
template Tensor<FLOAT16> Tensor<FLOAT16>::operator-(const TensorVariant& other) const;
template Tensor<FLOAT16> Tensor<FLOAT16>::operator*(const TensorVariant& other) const;

template Tensor<FLOAT32> Tensor<FLOAT32>::operator+(const TensorVariant& other) const;
template Tensor<FLOAT32> Tensor<FLOAT32>::operator-(const TensorVariant& other) const;
template Tensor<FLOAT32> Tensor<FLOAT32>::operator*(const TensorVariant& other) const;

template Tensor<INT8> Tensor<INT8>::operator+(const TensorVariant& other) const;
template Tensor<INT8> Tensor<INT8>::operator-(const TensorVariant& other) const;
template Tensor<INT8> Tensor<INT8>::operator*(const TensorVariant& other) const;

template Tensor<INT32> Tensor<INT32>::operator+(const TensorVariant& other) const;
template Tensor<INT32> Tensor<INT32>::operator-(const TensorVariant& other) const;
template Tensor<INT32> Tensor<INT32>::operator*(const TensorVariant& other) const;

template Tensor<UINT8> Tensor<UINT8>::operator+(const TensorVariant& other) const;
template Tensor<UINT8> Tensor<UINT8>::operator-(const TensorVariant& other) const;
template Tensor<UINT8> Tensor<UINT8>::operator*(const TensorVariant& other) const;

template Tensor<UINT32> Tensor<UINT32>::operator+(const TensorVariant& other) const;
template Tensor<UINT32> Tensor<UINT32>::operator-(const TensorVariant& other) const;
template Tensor<UINT32> Tensor<UINT32>::operator*(const TensorVariant& other) const;

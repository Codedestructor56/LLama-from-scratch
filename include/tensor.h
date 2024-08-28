#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <iostream>
#include <random>
#include <variant>
#include <numeric> 
#include <stdexcept>
#include <cuda_runtime.h>

typedef enum {
    FLOAT16,
    FLOAT32,
    INT8,
    INT32,
    UINT8,
    UINT32
} DType;

typedef enum{
  CUDA,
  CPU
} Device;

template<DType dtype>
struct DTypeToType;

template<>
struct DTypeToType<FLOAT16> { using Type = uint16_t; };

template<>
struct DTypeToType<FLOAT32> { using Type = float; };

template<>
struct DTypeToType<INT8> { using Type = int8_t; };

template<>
struct DTypeToType<INT32> { using Type = int32_t; };

template<>
struct DTypeToType<UINT8> { using Type = uint8_t; };

template<>
struct DTypeToType<UINT32> { using Type = uint32_t; };
  


template <DType dtype>
class Tensor;

using TensorVariant = std::variant<
        
        std::shared_ptr<Tensor<FLOAT32>>, 
        std::shared_ptr<Tensor<FLOAT16>>, 
        std::shared_ptr<Tensor<INT32>>, 
        std::shared_ptr<Tensor<UINT32>>, 
        std::shared_ptr<Tensor<INT8>>, 
        std::shared_ptr<Tensor<UINT8>>
    >;

extern size_t get_dtype_size(DType dtype);
extern void* allocate_memory(DType dtype, size_t num_elements);
extern void deallocate_memory(void* ptr);

template <DType dtype>
class Tensor : public std::enable_shared_from_this<Tensor<dtype>> {
    using T = typename DTypeToType<dtype>::Type;
    static_assert(
        std::is_same<T, float>::value ||
        std::is_same<T, uint16_t>::value ||
        std::is_same<T, int8_t>::value || 
        std::is_same<T, int32_t>::value || 
        std::is_same<T, uint8_t>::value || 
        std::is_same<T, uint32_t>::value, 
        "Unsupported type for Tensor"
    );

public:
    Tensor() : data_(nullptr), type(dtype), tens_device(CPU) {}

    Tensor(const std::vector<int>& shape) : type(dtype), shape(shape), tens_device(CPU) {
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        data_ = static_cast<T*>(allocate_memory(dtype, num_elems * sizeof(T)));
        for (int i = 0; i < num_elems; ++i) {
            data_[i] = 0;
        }
    }

    Tensor(T* data, const std::vector<int>& shape) 
        : type(dtype), data_(data), shape(shape), tens_device(CPU) {}

    Tensor(T* data, const std::vector<int>& shape, Device device) 
        : type(dtype), data_(data), shape(shape), tens_device(device) {}

    Tensor(std::vector<T>& vec, std::vector<int>& shape) 
        : type(dtype), shape(shape), tens_device(CPU) {
        initialize_from_vector(vec, shape);
    }

    Tensor(const std::vector<int>& vec, const std::vector<int>& shape) 
        : type(dtype), shape(shape), tens_device(CPU) {
        initialize_from_vector(vec, shape);
    }

    Tensor(const std::vector<float>& vec, std::vector<int>& shape) 
        : type(dtype), shape(shape), tens_device(CPU) {
        initialize_from_vector(vec, shape);
    }
    
    std::shared_ptr<Tensor<dtype>> shared_from_this() {
        return std::enable_shared_from_this<Tensor<dtype>>::shared_from_this();
    }

    std::shared_ptr<const Tensor<dtype>> shared_from_this() const{
        return std::enable_shared_from_this<Tensor<dtype>>::shared_from_this();
    }
  
    static Tensor<dtype> ones(const std::vector<int>& shape);
    static Tensor<dtype> zeros(const std::vector<int>& shape);
    static Tensor<dtype> rand(const std::vector<int>& shape);

    T get(const std::vector<int>& indices) const;
    Tensor<dtype> get_slice(const std::vector<int>& start_indices,
        const std::vector<int>& end_indices, const std::vector<int>& stride = {}) const;
    void set(const std::vector<int>& indices, const T& value);
    void set_slice(const std::vector<int>& start_indices, const std::vector<int>& end_indices,
        const std::vector<T>& values);
    void reshape(const std::vector<int>& new_shape);
    template<DType dt>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<dt>& tensor);
       
    template <DType new_dtype>
    std::shared_ptr<const Tensor<new_dtype>> change_dtype() const {
        auto new_tensor = std::make_shared<Tensor<new_dtype>>(shape);
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
         
        typename DTypeToType<new_dtype>::Type* new_data = new typename DTypeToType<new_dtype>::Type[num_elems];
 
        for (int i = 0; i < num_elems; ++i) {
            new_data[i] = static_cast<typename DTypeToType<new_dtype>::Type>(data_[i]);
        }

        new_tensor->data_set(new_data);
        new_tensor->set_children(this->children);
        return new_tensor;
    }

    size_t get_children_size() const {
        return children.size();
    } 
    
    const std::vector<int> get_shape(){
      return shape;
    }
    const std::vector<TensorVariant>& get_children() const {
        return children;
    }

    void set_children(const std::vector<TensorVariant>& children) {
        this->children = children; 
    }

    Device get_device() const{
      return tens_device;
    }

    void change_device(const Device& device){
      tens_device = device;
    }

    Tensor<dtype> operator+(const Tensor<dtype>& other) const;
    Tensor<dtype> operator-(const Tensor<dtype>& other) const;
    Tensor<dtype> operator*(const Tensor<dtype>& other) const; 
    
    Tensor<dtype> operator+(const TensorVariant& other) const;
    Tensor<dtype> operator-(const TensorVariant& other) const;
    Tensor<dtype> operator*(const TensorVariant& other) const;
    
       
    int size() const {
      int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      return size;
    }


    T* data() const { return data_; } 
    void data_set(const T* data) { data_ = const_cast<T*>(data);}
    std::vector<int> shape;
    DType type;
    std::shared_ptr<Tensor> grad;

private:
    Device tens_device;
    template <typename Op>
    Tensor<dtype> tensorOperation(const TensorVariant& rhs, Op op) const;
    T* data_;  
    std::vector<TensorVariant> children;

    void allocate_and_initialize(const std::vector<int>& shape, bool zero_initialize, bool is_rand);

    template <typename VecType>
    void initialize_from_vector(const std::vector<VecType>& vec, const std::vector<int>& shape) {
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (num_elems != vec.size()) {
            throw std::runtime_error("Shape does not match the number of elements in vector");
        }
        data_ = static_cast<T*>(allocate_memory(dtype, vec.size()));
        for (size_t i = 0; i < vec.size(); ++i) {
            data_[i] = static_cast<T>(vec[i]);
        }
    }
};


template <typename T, typename Op>
void tensorOperationCuda(const T* a, const T* b, T* result, int num_elems, Op op, int block_size);
template <typename T>
void matmul_cuda(const T* A, const T* B, T* C, int m, int n, int p); 
//defining it outside the class
template<DType dtype>
extern Tensor<dtype> matmul(const Tensor<dtype>& tens1, const Tensor<dtype>& tens2);

void atomicMulTensor(Tensor<FLOAT32>& tensor, const Tensor<FLOAT32>& values);

template<DType dtype>
extern Tensor<dtype> hstack(const std::vector<Tensor<dtype>>& tensors);

template<DType dtype>
extern Tensor<dtype> hstack(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2);

template<DType dtype>
extern Tensor<dtype> vstack(const std::vector<Tensor<dtype>>& tensors);

template <DType dtype>
Tensor<dtype> vstack(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2); 
// Explicit template instantiation
template class Tensor<FLOAT16>;
template class Tensor<FLOAT32>;
template class Tensor<INT8>;
template class Tensor<INT32>;
template class Tensor<UINT8>;
template class Tensor<UINT32>;


#endif

#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <iostream>
#include <random>

typedef enum {
    FLOAT16,
    FLOAT32,
    INT8,
    INT32,
    UINT8,
    UINT32
} DType;


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
    Tensor() : data_(nullptr), type(dtype) {}
    Tensor(T* data, std::vector<int>& shape): type(dtype), data_(data), shape(shape){}
    Tensor(std::vector<T>& vec, std::vector<int>& shape): type(dtype), shape(shape){
        data_ = static_cast<T*>(allocate_memory(dtype, vec.size()));
        for (size_t i = 0; i < vec.size(); ++i) {
            data_[i] = vec[i];
        }
    }

    static Tensor<dtype> ones(const std::vector<int>& shape);
    static Tensor<dtype> zeros(const std::vector<int>& shape);
    static Tensor<dtype> rand(const std::vector<int>& shape);

    T get(const std::vector<int>& indices) const;
    Tensor<dtype> get_slice(const std::vector<int>& start_indices,
        const std::vector<int>& end_indices, int traversal_strategy = 1, const std::vector<int>& stride = {}) const;
    void set(const std::vector<int>& indices, const T& value);
    void set_slice(const std::vector<int>& start_indices, const std::vector<int>& end_indices, const Tensor<dtype>& values);
    template<DType dt>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<dt>& tensor);

    Tensor<dtype> operator+(const Tensor<dtype>& other)const;
    Tensor<dtype> operator-(const Tensor<dtype>& other)const;
    Tensor<dtype> operator*(const Tensor<dtype>& other)const; 

    T* data() const { return data_; } 

    std::vector<int> shape;
    DType type;
    std::shared_ptr<Tensor> grad;

private:
    T* data_;  
    std::vector<std::shared_ptr<Tensor<dtype>>> children;

    void allocate_and_initialize(const std::vector<int>& shape, bool zero_initialize, bool is_rand);
};
//defining it outside the class
template<DType dtype>
Tensor<dtype> matmul(const Tensor<dtype>& tens1, const Tensor<dtype>& tens2);
#endif

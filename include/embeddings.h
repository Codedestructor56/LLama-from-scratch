#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "tensor.h"
#include <memory>
#include <vector>
#include <random>

template<DType dtype>
class Embeddings {
public: 
    Embeddings(size_t vocab_size, size_t embedding_dim);
    Tensor<dtype> forward(const Tensor<UINT32>& input);

    Tensor<dtype> backward(const Tensor<dtype>& grad_output);
    void update(const Tensor<dtype>& grad, float learning_rate);

    Tensor<dtype> get_embedding_matrix() const  {
      return embedding_matrix_;
    }

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    Tensor<dtype> embedding_matrix_;
};

template<DType dtype>
Embeddings<dtype>::Embeddings(size_t vocab_size, size_t embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim), 
      embedding_matrix_(Tensor<dtype>::rand({static_cast<int>(vocab_size), static_cast<int>(embedding_dim)})) {}

template<DType dtype>
Tensor<dtype> Embeddings<dtype>::forward(const Tensor<UINT32>& input) {
    std::vector<int> output_shape = {static_cast<int>(input.shape[0]), static_cast<int>(embedding_dim_)};
    Tensor<dtype> output = Tensor<dtype>::zeros(output_shape);
    
    for (size_t i = 0; i < input.shape[0]; ++i) {
        int token_id = static_cast<int>(input.get({static_cast<int>(i), 0}));
        if(token_id >= vocab_size_){
          throw std::out_of_range("Token Ids should not exceed vocab size");
        }
        for (size_t j = 0; j < embedding_dim_; ++j) {
            output.set({static_cast<int>(i), static_cast<int>(j)}, embedding_matrix_.get({token_id, static_cast<int>(j)}));
        }
    }
    return output;
}

#endif


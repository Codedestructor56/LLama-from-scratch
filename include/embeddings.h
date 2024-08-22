#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "tensor.h"
#include <memory>
#include <vector>

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


#endif 

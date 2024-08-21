#include "embeddings.h"
#include <random>

template<DType dtype>
Embeddings<dtype>::Embeddings(size_t vocab_size, size_t embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim), 
      embedding_matrix_(Tensor<dtype>::rand({vocab_size, embedding_dim})) {}


//since this module will only interact with the dataloader, I am assuming that only
//tensors of shape (batch_size, 1) will be passed to this module
template<DType dtype>
Tensor<dtype> Embeddings<dtype>::forward(const Tensor<UINT32>& input) {
    std::vector<int> output_shape = {static_cast<int>(input.shape[0]), static_cast<int>(embedding_dim_)};
    Tensor<dtype> output(output_shape);
    
    for (int i = 0; i < input.shape[0]; ++i) {
        auto token_id = input.get({i, 0});
        for (int j = 0; j < embedding_dim_; ++j) {
            output.set({i, j}, embedding_matrix_.get({token_id, j}));
        }
    }
    return output;
}
template class Embeddings<FLOAT16>;
template class Embeddings<FLOAT32>;
template class Embeddings<INT8>;
template class Embeddings<INT32>;
template class Embeddings<UINT8>;
template class Embeddings<UINT32>;

#endif 

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "tensor.h"  
#include <unordered_map>
#include <string>
#include <vector>

template<DType dtype>
class Embeddings {
public:
    Embeddings(size_t vocab_size, size_t embedding_dim);

    Tensor<dtype> initialize_from_tokens(const Tensor<std::string>& tokens);

    void set_embedding(const std::string& token, const std::vector<T>& embedding);

    std::vector<T> get_embedding(const std::string& token) const;

private:
    size_t embedding_dim_;
    std::unordered_map<std::string, std::vector<T>> embedding_map_;
};

#endif

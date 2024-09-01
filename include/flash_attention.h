#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include "tensor.h"
#include "rms_norm.h"

template<DType dtype>
class FlashAttention {
public:
    FlashAttention(int head_dim, int num_heads, Device device = CPU);

    Tensor<dtype> forward(const Tensor<dtype>& query, const Tensor<dtype>& key, const Tensor<dtype>& value);
    Tensor<dtype> backward(const Tensor<dtype>& grad_output);

private:
    Tensor<dtype> softmax(const Tensor<dtype>& input);
    Tensor<dtype> matmul(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2);

    int head_dim_;
    int num_heads_;
    Device device_;
    
    Tensor<dtype> attention_scores_;
    Tensor<dtype> attention_probs_;
    Tensor<dtype> context_;
};

template<Dtype dtype>
FlashAttention::FlashAttention(int head_dim_, int num_heads_, Device device)
  :head_dim_(head_dim_), num_heads_(num_heads_),device(device){}


#endif 

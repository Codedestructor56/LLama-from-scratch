#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "dataloader.h"
#include "embeddings.h"
#include <memory>

template<DType dtype>
class RMSNorm {
public:
    RMSNorm(std::unique_ptr<Dataloader> dataloader, std::unique_ptr<Embeddings<dtype>> embeddings, float epsilon, Device device = CPU);

    Tensor<dtype> forward(const Tensor<dtype>& input);
    Tensor<dtype> backward(const Tensor<dtype>& grad_output);

private:
    std::unique_ptr<Dataloader> dataloader_;
    std::unique_ptr<Embeddings<dtype>> embeddings_;
    float epsilon_;
    Device device_;
};

template<DType dtype>
RMSNorm<dtype>::RMSNorm(std::unique_ptr<Dataloader> dataloader, std::unique_ptr<Embeddings<dtype>> embeddings, float epsilon, Device device)
: dataloader_(std::move(dataloader)), embeddings_(std::move(embeddings)), epsilon_(epsilon), device_(device) {}


template class RMSNorm<FLOAT32>;
template class RMSNorm<FLOAT16>;
template class RMSNorm<INT8>;
template class RMSNorm<INT32>;
template class RMSNorm<UINT8>;
template class RMSNorm<UINT32>;
#endif

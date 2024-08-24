#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "dataloader.h"
#include "embeddings.h"
#include <memory>

template<DType dtype>
class RMSNorm {
public:
    RMSNorm(std::unique_ptr<Dataloader> dataloader, std::unique_ptr<Embeddings<dtype>> embeddings,
        float epsilon);

    Tensor<dtype> forward(const Tensor<dtype>& input);
    Tensor<dtype> backward(const Tensor<dtype>& grad_output);

private:
    std::unique_ptr<Dataloader> dataloader_;
    std::unique_ptr<Embeddings<dtype>> embeddings_;
    float epsilon_;
};

#endif

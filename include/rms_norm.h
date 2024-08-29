#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "dataloader.h"
#include "embeddings.h"
#include <memory>
#include <numeric>
#include <cmath>
#include <vector>

template<DType dtype>
class RMSNorm {
public:
    RMSNorm(std::shared_ptr<Dataloader> dataloader, std::shared_ptr<Embeddings<dtype>> embeddings, float epsilon, Device device = CPU);

    Tensor<dtype> forward(const Tensor<dtype>& input);
    Tensor<dtype> backward(const Tensor<dtype>& grad_output);

private:
    std::shared_ptr<Dataloader> dataloader_;
    std::shared_ptr<Embeddings<dtype>> embeddings_;
    float epsilon_;
    Device device_;
};

template<DType dtype>
RMSNorm<dtype>::RMSNorm(std::shared_ptr<Dataloader> dataloader, std::shared_ptr<Embeddings<dtype>> embeddings, float epsilon, Device device)
: dataloader_(std::move(dataloader)), embeddings_(std::move(embeddings)), epsilon_(epsilon), device_(device) {}

template<DType dtype>
Tensor<dtype> RMSNorm<dtype>::forward(const Tensor<dtype>& input) { 
    using T = typename DTypeToType<dtype>::Type;
    Tensor<UINT32> batch_data = dataloader_->get_next_batch_uint32();
    Tensor<dtype> embedded_data = embeddings_->forward(batch_data);
    
    std::vector<int> shape = embedded_data.get_shape();
    int num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    Tensor<dtype> normed_tensor(shape);
    T mean_square = 0;
    for (int i = 0; i < num_elements; ++i) {
        mean_square += embedded_data.data()[i] * embedded_data.data()[i];
    }
    mean_square /= num_elements;
    T rms = std::sqrt(mean_square + epsilon_);
    for (int i = 0; i < num_elements; ++i) {
        normed_tensor.data()[i] = embedded_data.data()[i] / rms;
    }

    return normed_tensor;
}

#endif  

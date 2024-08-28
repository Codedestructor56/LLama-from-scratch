#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "dataloader.h"
#include "embeddings.h"
#include <memory>
#include <numeric>
#include <cmath>
#include <vector>

// Ensure DType enum is defined elsewhere
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

template<DType dtype>
Tensor<dtype> RMSNorm<dtype>::forward(const Tensor<dtype>& input) {
    // Retrieve the next batch and forward through embeddings
    using T = typename DTypeToType<dtype>::Type;
    Tensor<UINT32> batch_data = dataloader_->get_next_batch_uint32();
    Tensor<dtype> embedded_data = embeddings_->forward(batch_data);
    
    // Get shape and compute the number of elements
    std::vector<int> shape = embedded_data.get_shape();
    int num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    // Prepare output tensor
    Tensor<dtype> normed_tensor(shape);

    // Compute mean square of input values
    T mean_square = 0;
    for (int i = 0; i < num_elements; ++i) {
        mean_square += embedded_data.data()[i] * embedded_data.data()[i];
    }
    mean_square /= num_elements;

    // Compute root mean square and normalize
    T rms = std::sqrt(mean_square + epsilon_);
    for (int i = 0; i < num_elements; ++i) {
        normed_tensor.data()[i] = embedded_data.data()[i] / rms;
    }

    return normed_tensor;
}

// You may need explicit template instantiations if using a separate implementation file
// Example:
// template class RMSNorm<FLOAT32>;
// template class RMSNorm<FLOAT16>;
// template class RMSNorm<INT8>;
// template class RMSNorm<INT32>;
// template class RMSNorm<UINT8>;
// template class RMSNorm<UINT32>;

#endif  // RMS_NORM_H

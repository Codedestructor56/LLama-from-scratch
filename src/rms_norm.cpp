#include "rms_norm.h"

template<DType dtype>
RMSNorm<dtype>::RMSNorm(std::unique_ptr<Dataloader> dataloader, std::unique_ptr<Embeddings<dtype>> embeddings, float epsilon)
    : dataloader_(std::move(dataloader)), embeddings_(std::move(embeddings)), epsilon_(epsilon) {}

template<DType dtype>
Tensor<dtype> RMSNorm<dtype>::forward(const Tensor<dtype>& input) {
    Tensor<UINT32> batch_data = dataloader_->get_next_batch_uint32();
    
    Tensor<dtype> embedded_data = embeddings_->forward(batch_data);
    std::vector<int> shape = embedded_data.shape;
    int num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    dtype* data = embedded_data.data();
    Tensor<dtype> normed_tensor(shape);
    dtype* normed_data = normed_tensor.data();

    dtype mean_square = 0;
    for (int i = 0; i < num_elements; ++i) {
        mean_square += data[i] * data[i];
    }
    mean_square /= num_elements;
   
    dtype rms = std::sqrt(mean_square + epsilon_);
    for (int i = 0; i < num_elements; ++i) {
        normed_data[i] = data[i] / rms;
    }

    return normed_tensor;
}

template class RMSNorm<FLOAT32>;
template class RMSNorm<FLOAT16>;
template class RMSNorm<INT8>;
template class RMSNorm<INT32>;
template class RMSNorm<UINT8>;
template class RMSNorm<UINT32>;

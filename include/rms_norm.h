#ifndef RMS_NORM_H
#define RMS_NORM_H

#include <memory>
#include <numeric>
#include <cmath>
#include <vector>

template<DType dtype>
class RMSNorm {
public:
    RMSNorm(float epsilon, Device device = CPU);

    Tensor<dtype> forward(const Tensor<dtype>& input);
    Tensor<dtype> backward(const Tensor<dtype>& grad_output);

private:
    float epsilon_;
    Device device_;
};

template<DType dtype>
RMSNorm<dtype>::RMSNorm(float epsilon, Device device)
: epsilon_(epsilon), device_(device) {}

template<DType dtype>
Tensor<dtype> RMSNorm<dtype>::forward(const Tensor<dtype>& input) { 
    using T = typename DTypeToType<dtype>::Type;

    std::vector<int> shape = input.get_shape();
    int num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    Tensor<dtype> normed_tensor(shape);
    T mean_square = 0;
    for (int i = 0; i < num_elements; ++i) {
        mean_square += input.data()[i] * input.data()[i];
    }
    mean_square /= num_elements;
    T rms = std::sqrt(mean_square + epsilon_);
    for (int i = 0; i < num_elements; ++i) {
        normed_tensor.data()[i] = input.data()[i] / rms;
    }

    return normed_tensor;
}

#endif

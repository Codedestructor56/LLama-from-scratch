#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

class Dataloader {
public:
    Dataloader(const std::string& data_path, size_t batch_size);
    ~Dataloader();
 
    void start_loading();

    void stop_loading();

    Tensor<FLOAT32> get_next_batch_float32();
    Tensor<FLOAT16> get_next_batch_float16();
    Tensor<INT8> get_next_batch_int8();
    Tensor<INT32> get_next_batch_int32();
    Tensor<UINT8> get_next_batch_uint8();
    Tensor<UINT32> get_next_batch_uint32();

    Tensor<UINT32> text_to_tensor_ids(const std::string& text, const std::vector<int>& shape);

private:
     void load_data();

    Tensor<FLOAT32> vector_to_tensor_float32(const std::vector<float>& vec, const std::vector<int>& shape);
    Tensor<FLOAT16> vector_to_tensor_float16(const std::vector<uint16_t>& vec, const std::vector<int>& shape);
    Tensor<INT8> vector_to_tensor_int8(const std::vector<int8_t>& vec, const std::vector<int>& shape);
    Tensor<INT32> vector_to_tensor_int32(const std::vector<int32_t>& vec, const std::vector<int>& shape);
    Tensor<UINT8> vector_to_tensor_uint8(const std::vector<uint8_t>& vec, const std::vector<int>& shape);
    Tensor<UINT32> vector_to_tensor_uint32(const std::vector<uint32_t>& vec, const std::vector<int>& shape);

    void convert_to_tensor();
    void process_data();
    void distribute_tasks();
    void gather_results();

    sentencepiece::SentencePieceProcessor tokenizer_;
    std::string data_path_;
    size_t batch_size_;
    bool stop_requested_;
    std::vector<std::thread> worker_threads_;
    std::mutex data_mutex_;
    TensorVariant current_batch_;
};

#endif

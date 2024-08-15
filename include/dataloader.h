#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <variant>
#include <fstream>
#include <sstream>
#include <condition_variable>
#include <sentencepiece_processor.h>
#include <type_traits>

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
    void process_file_segment(size_t start, size_t end);

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
    std::condition_variable cv_; 
    std::vector<std::string> words;
    TensorVariant current_batch_;
    size_t current_word_index_; 
};

#endif

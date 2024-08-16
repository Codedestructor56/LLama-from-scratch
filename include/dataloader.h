#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <string>
#include <vector>
#include <numeric> 
#include <stdexcept>
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <variant>
#include <fstream>
#include <sstream>
#include <condition_variable>
#include <queue>
#include <sentencepiece_processor.h>
#include <type_traits>


class Dataloader {
public:
    Dataloader(const std::string& data_path, size_t batch_size);
    
    Tensor<FLOAT32> get_next_batch_float32();
    Tensor<FLOAT16> get_next_batch_float16();
    Tensor<INT8> get_next_batch_int8();
    Tensor<INT32> get_next_batch_int32();
    Tensor<UINT8> get_next_batch_uint8();
    Tensor<UINT32> get_next_batch_uint32();

    void start_loading();
    void stop_loading();

private:
    void load_data();
    void background_load_data();

    sentencepiece::SentencePieceProcessor tokenizer_;
    std::string data_path_;
    size_t batch_size_;
    std::vector<std::string> words;
    TensorVariant current_batch_;
    size_t current_word_index_;
    bool done_;

    std::queue<TensorVariant> batch_queue_;
    std::thread loader_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
    bool is_loading_done_;
};

#endif

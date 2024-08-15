#include "dataloader.h"
#include <numeric> 
#include <stdexcept>
#include <iostream>

TensorVariant wrapInVariant(const auto& batch) {
    using T = std::decay_t<decltype(batch)>;
    if constexpr (std::is_same_v<T, std::shared_ptr<Tensor<FLOAT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<FLOAT16>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<INT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<UINT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<INT8>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<UINT8>>>) {
        return TensorVariant(batch);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

Dataloader::Dataloader(const std::string& data_path, size_t batch_size)
    : data_path_(data_path),
      batch_size_(batch_size),
      current_word_index_(0),
      stop_requested_(false) {
   
    std::string model_path = "./bpe_tokenizer.model";
    if (!tokenizer_.Load(model_path).ok()) {
        throw std::runtime_error("Failed to load SentencePiece model from " + model_path);
    }
     
    std::ifstream file("./data/" + data_path_ + ".txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << "/data/" + data_path_ + ".txt" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) { 
            words.push_back(word);
        }
    }

    file.close();

    distribute_tasks();
}

Dataloader::~Dataloader() {
    stop_loading(); 
}

void Dataloader::distribute_tasks() {
    size_t num_threads = std::thread::hardware_concurrency();
    size_t segment_size = words.size() / num_threads; 
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * segment_size;
        size_t end = (i == num_threads - 1) ? words.size() : (i + 1) * segment_size;
        worker_threads_.emplace_back(&Dataloader::process_file_segment, this, start, end);
    }
}

void Dataloader::process_file_segment(size_t start, size_t end) {
    std::vector<Tensor<UINT32>> local_tensors;
    size_t local_batch_size = 0;

    for (size_t i = start; i < end; ++i) {
        if (stop_requested_) break;

        std::string word = words[i];
        std::vector<int> ids;
        tokenizer_.Encode(word, &ids);

        for (int token_id : ids) {
            Tensor<UINT32> tensor = Tensor<UINT32>({token_id}, {1});
            local_tensors.push_back(tensor);
            local_batch_size++;

            if (local_batch_size == batch_size_) {
                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    Tensor<UINT32> stacked_tensors = vstack(local_tensors);
                    current_batch_ = wrapInVariant(std::make_shared<Tensor<UINT32>>(stacked_tensors));
                }
                local_tensors.clear();
                local_batch_size = 0;
                cv_.notify_one();
            }
        }
    }

    // Handle any remaining tensors that didn't form a full batch
    if (!local_tensors.empty()) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        Tensor<UINT32> stacked_tensors = vstack(local_tensors);
        current_batch_ = wrapInVariant(std::make_shared<Tensor<UINT32>>(stacked_tensors));
        cv_.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        stop_requested_ = true;
    }
    cv_.notify_one();
}

void Dataloader::start_loading() {
    if (worker_threads_.empty()) {
        distribute_tasks();
    }
}

void Dataloader::stop_loading() {
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();

    for (std::thread &worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    worker_threads_.clear();
}

Tensor<UINT32> Dataloader::get_next_batch_uint32() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    cv_.wait(lock, [this]() { return !std::holds_alternative<std::monostate>(current_batch_) || stop_requested_; });

    if (stop_requested_) {
        return Tensor<UINT32>();  
    }

    if (auto ptr = std::get_if<std::shared_ptr<Tensor<UINT32>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<FLOAT16> Dataloader::get_next_batch_float16() {
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<FLOAT16>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<INT8> Dataloader::get_next_batch_int8() {
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<INT8>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<INT32> Dataloader::get_next_batch_int32() {
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<INT32>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<UINT8> Dataloader::get_next_batch_uint8() {
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<UINT8>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

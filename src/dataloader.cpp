#include "dataloader.h"



Dataloader::Dataloader(const std::string& data_path, size_t batch_size)
    : data_path_(data_path),
      batch_size_(batch_size),
      current_word_index_(0),
      done_(false),
      stop_(false),
      is_loading_done_(false) {

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
}

void Dataloader::start_loading() {
    stop_ = false;
    loader_thread_ = std::thread(&Dataloader::background_load_data, this);
}

void Dataloader::stop_loading() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (loader_thread_.joinable()) {
        loader_thread_.join();
    }
}

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


void Dataloader::background_load_data() {
    while (!stop_) {
        load_data();

        if (current_word_index_ >= words.size()) {
            std::lock_guard<std::mutex> lock(mutex_);
            is_loading_done_ = true;
            cv_.notify_all();
            break;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch_queue_.push(current_batch_);
        }

        cv_.notify_all();
    }
}


void Dataloader::load_data() {
    std::vector<Tensor<UINT32>> tensors;
    Tensor<UINT32> stacked_tensors;
    size_t current_batch_size = 0;

    while (current_word_index_ < words.size() && current_batch_size < batch_size_) {
        std::string word = words[current_word_index_];
        current_word_index_++; 

        std::vector<int> ids;
        tokenizer_.Encode(word, &ids); 

        for (int token_id : ids) {
            Tensor<UINT32> tensor = Tensor<UINT32>({token_id}, {1});
            tensors.push_back(tensor);
            current_batch_size++;

            if (current_batch_size >= batch_size_) {
                break; 
            }
        }
    }
    
    if (!tensors.empty()) {
        stacked_tensors = vstack(tensors);
        current_batch_ = wrapInVariant(std::make_shared<Tensor<UINT32>>(stacked_tensors));
        tensors.clear();
    } else { 
        current_batch_ = TensorVariant(); 
    }
}

Tensor<FLOAT32> Dataloader::get_next_batch_float32() {
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<FLOAT32>>>(&current_batch_)) {
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


Tensor<UINT32> Dataloader::get_next_batch_uint32() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !batch_queue_.empty() || is_loading_done_; });

    if (batch_queue_.empty()) {
        return Tensor<UINT32>();
    }

    TensorVariant batch = batch_queue_.front();
    batch_queue_.pop();

    if (auto ptr = std::get_if<std::shared_ptr<Tensor<UINT32>>>(&batch)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}




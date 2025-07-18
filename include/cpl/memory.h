#pragma once

#include "cpl/common.h"

namespace cpl {

template<typename T>
class DeviceMemory {
public:
    // Default constructor
    DeviceMemory() : data_(nullptr), size_(0) {}

    // Constructor that allocates memory
    explicit DeviceMemory(size_t size) : data_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
        }
    }

    // Destructor that frees memory
    ~DeviceMemory() {
        if (data_) {
            // It's good practice to not throw from a destructor.
            // CUDA_CHECK can exit(), which might be okay for this library's goals.
            // Or we can log the error. For now, we stick with CUDA_CHECK.
            CUDA_CHECK(cudaFree(data_));
        }
    }

    // Disable copy constructor and copy assignment
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move constructor and move assignment
    DeviceMemory(DeviceMemory&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (data_) {
                CUDA_CHECK(cudaFree(data_));
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Getter for raw pointer
    T* get() { return data_; }
    const T* get() const { return data_; }

    // Getter for size
    size_t size() const { return size_; }
    
    // Getter for size in bytes
    size_t bytes() const { return size_ * sizeof(T); }

private:
    T* data_;
    size_t size_;
};

} // namespace cpl 
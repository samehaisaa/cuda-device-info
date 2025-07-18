#pragma once

#include <vector>
#include "cpl/common.h"
#include "cpl/memory.h"
#include "cpl/logging.h"

namespace cpl {

template<typename T>
class Vector {
public:
    explicit Vector(size_t size) : size_(size), host_data_(size) {}

    size_t size() const { return size_; }
    T& at(size_t i) { return host_data_.at(i); }
    const T& at(size_t i) const { return host_data_.at(i); }
    T* data() { return host_data_.data(); }
    const T* data() const { return host_data_.data(); }

    void to_device(DeviceMemory<T>& device_mem) const {
        if (device_mem.size() != size_) {
            logging::log_error("Device memory size does not match vector size for to_device.");
            return;
        }
        CUDA_CHECK(cudaMemcpy(device_mem.get(), data(), device_mem.bytes(), cudaMemcpyHostToDevice));
    }

    void from_device(const DeviceMemory<T>& device_mem) {
        if (device_mem.size() != size_) {
            logging::log_error("Device memory size does not match vector size for from_device.");
            return;
        }
        CUDA_CHECK(cudaMemcpy(data(), device_mem.get(), device_mem.bytes(), cudaMemcpyDeviceToHost));
    }

private:
    size_t size_;
    std::vector<T> host_data_;
};

} // namespace cpl

void vector_add(const cpl::Vector<float>& A, const cpl::Vector<float>& B, cpl::Vector<float>& C);
void initializeVector(cpl::Vector<float>& vec);
void verify_vector_addition(const cpl::Vector<float>& A, const cpl::Vector<float>& B, const cpl::Vector<float>& C);
void vector_subtract(const cpl::Vector<float>& A, const cpl::Vector<float>& B, cpl::Vector<float>& C);
float dot_product(const cpl::Vector<float>& A, const cpl::Vector<float>& B);

#pragma once

#include <vector>
#include "cpl/common.h"
#include "cpl/memory.h"
#include "cpl/logging.h"
#include "cpl/vector.h"

namespace cpl {

template<typename T>
class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), host_data_(rows * cols) {}

    T& at(int r, int c) {
        if (r >= rows_ || c >= cols_) {
            logging::log_error("Matrix access out of bounds");
            // This is not great, but for now it's a placeholder
            throw std::out_of_range("Matrix access out of bounds");
        }
        return host_data_[r * cols_ + c];
    }
    const T& at(int r, int c) const {
        if (r >= rows_ || c >= cols_) {
            logging::log_error("Matrix access out of bounds");
            throw std::out_of_range("Matrix access out of bounds");
        }
        return host_data_[r * cols_ + c];
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t size() const { return host_data_.size(); }
    T* data() { return host_data_.data(); }
    const T* data() const { return host_data_.data(); }

    // Copy data to a device memory object.
    void to_device(DeviceMemory<T>& device_mem) const {
        if (device_mem.size() != size()) {
            logging::log_error("Device memory size does not match matrix size for to_device.");
            return;
        }
        CUDA_CHECK(cudaMemcpy(device_mem.get(), data(), device_mem.bytes(), cudaMemcpyHostToDevice));
    }

    // Copy data from a device memory object
    void from_device(const DeviceMemory<T>& device_mem) {
        if (device_mem.size() != size()) {
            logging::log_error("Device memory size does not match matrix size for from_device.");
            return;
        }
        CUDA_CHECK(cudaMemcpy(data(), device_mem.get(), device_mem.bytes(), cudaMemcpyDeviceToHost));
    }

private:
    int rows_;
    int cols_;
    std::vector<T> host_data_;
};

} // namespace cpl


// To be refactored to use the Matrix class
void matrix_multiply(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C);

void print_matrix(const cpl::Matrix<float>& matrix, const char* name);

void element_wise_add(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C);
void matrix_vector_multiply(const cpl::Matrix<float>& A, const cpl::Vector<float>& x, cpl::Vector<float>& y);
void element_wise_subtract(const cpl::Matrix<float>& A, const cpl::Matrix<float>& B, cpl::Matrix<float>& C);
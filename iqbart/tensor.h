
#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>
#include <cstddef>  // size_t
#include <stdexcept>  // std::invalid_argument
#include <initializer_list>  // For braced init/reshape

template <typename T>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    // Reusable stride computation (row-major, no resize)
    void compute_strides() {
        if (shape_.empty()) { return; }
        strides_.resize(shape_.size());
        strides_.back() = 1;
        for (size_t i = shape_.size() - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
    }

    // Reusable total size calc
    size_t total_size_impl(const std::vector<size_t>& dims) const {
        size_t total = 1;
        for (auto dim : dims) { total *= dim; }
        return total;
    }

    template <typename... Args>
    size_t get_offset(Args... args) const {
        const size_t indices[] = {static_cast<size_t>(args)...};
        size_t offset = 0;
        if (shape_.empty()) { return 0; }
        size_t num_indices = sizeof...(args);
        for (size_t i = 0; i < std::min(num_indices, shape_.size()); ++i) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

public:
    // Default: Empty Tensor
    Tensor() = default;

    // Modern braced init for shape (e.g., Tensor<double>{nkeeptrain, n} or {} for empty)
    Tensor(std::initializer_list<size_t> ilist) : shape_(ilist) {
        compute_strides();
        data_.resize(total_size_impl(shape_), T{});
    }

    // Move ctor (efficient transfer)
    Tensor(Tensor&& other) noexcept
        : data_(std::move(other.data_)), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)) {}

    // Copy ctor (implicit from vector/shape, but explicit for clarity)
    Tensor(const Tensor& other)
        : data_(other.data_), shape_(other.shape_), strides_(other.strides_) {
        if (!other.shape_.empty()) {
            compute_strides();
        }
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        return *this;
    }

    // Copy assignment (implicit, but explicit for completeness)
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data_ = other.data_;
            shape_ = other.shape_;
            strides_ = other.strides_;
            if (!shape_.empty()) {
                compute_strides();  // Recompute if copied changed strides
            }
        }
        return *this;
    }

    // NumPy-compatible reshape (vector overload)
    void reshape(std::vector<size_t> new_shape) {
        size_t new_total = total_size_impl(new_shape);
        if (new_total != data_.size()) {
            throw std::invalid_argument("New shape total size must match current data size");
        }
        shape_ = std::move(new_shape);
        compute_strides();  // Update strides for new shape
    }

    // Braced reshape (e.g., reshape({np, l_qp}))
    template <typename... Sizes>
    void reshape(std::initializer_list<size_t> ilist) {
        std::vector<size_t> new_shape(ilist);
        reshape(std::move(new_shape));
    }

    // Unchecked ND access
    template <typename... Args>
    T& operator()(Args... args) {
        return data_[get_offset(args...)];
    }

    // Const access
    template <typename... Args>
    const T& operator()(Args... args) const {
        const size_t offset = get_offset(args...);
        return data_[offset];  // Direct const access (safer than const_cast)
    }

    // Convenience: 1D index for any ndim (treat as flat)
    T& operator[](size_t i) { return operator()(i); }
    const T& operator[](size_t i) const { return operator()(i); }

    // Accessors
    bool empty() const { return data_.empty(); }  // For your if (!empty())
    size_t size() const { return data_.size(); }  // Total elements
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
};

#endif // TENSOR_H_

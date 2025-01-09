#include "../include/tensor3d.hpp"

Tensor3D::Tensor3D() : height(0), width(0), depth(0) {}

// legacy constructor for 2D matrix compatibility 
Tensor3D::Tensor3D(size_t rows, size_t cols) : depth(1), height(rows), width(cols) { 
    data.resize(depth * height * width); 
}

// construct from dimensions
Tensor3D::Tensor3D(size_t depth, size_t height, size_t width) : height(height), width(width), depth(depth) {
    data.resize(depth * height * width);
}

// construct from flat vector
Tensor3D::Tensor3D(size_t depth, size_t height, size_t width, const std::vector<float> &data)
    : height(height), width(width), depth(depth) {
    if (data.size() != depth * height * width) {
        throw std::invalid_argument("data length and dimension mismatch");
    }
    this->data = data;
}

// construct from 3D vector
Tensor3D::Tensor3D(size_t depth, size_t height, size_t width, const std::vector<std::vector<std::vector<float>>> &data)
    : height(height), width(width), depth(depth) {
    if (data.size() != depth or data[0].size() != height or data[0][0].size() != width) {
        throw std::invalid_argument("data length and dimension mismatch");
    }
    // flatten 3D vector into 1D storage
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                this->data[d * height * width + h * width + w] = data[d][h][w];
            }
        }
    }
}

// compute linear index from 3D coordinates
size_t Tensor3D::index(size_t d, size_t h, size_t w) { return d * (height * width) + h * (width) + w; }
const size_t Tensor3D::index(size_t d, size_t h, size_t w) const { return d * (height * width) + h * (width) + w; }

// access elements using 3D coordinates
float &Tensor3D::operator()(size_t d, size_t h, size_t w) {
    if (d >= depth or h >= height or w >= width) {
        throw std::runtime_error("index out of range in Tensor3D::operator()");
    }
    return data[index(d, h, w)];
}
const float &Tensor3D::operator()(size_t d, size_t h, size_t w) const {
    if (d >= depth or h >= height or w >= width) {
        throw std::runtime_error("index out of range in Tensor3D::operator()");
    }
    return data[index(d, h, w)];
}

// extract 2D slice at given depth
Tensor3D Tensor3D::operator()(size_t d) {
    size_t slice_size = width * height;
    std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
    return Tensor3D(1, height, width, new_data);
}

const Tensor3D Tensor3D::operator()(size_t d) const {
    size_t slice_size = width * height;
    std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
    return Tensor3D(1, height, width, new_data);
}

std::vector<float>& Tensor3D::get_flat_data() { return data; }

const std::vector<float>& Tensor3D::get_flat_data() const { return data; }

// compute dot product with a kernel centered at specific position - the argument must be the kernel
float Tensor3D::dot_with_kernel_at_position(const Tensor3D &kernel, size_t start_x, size_t start_y) const {
    float sum = 0.0;

    // to facilitate start_x and start_y being the centre position
    int kernel_width_offset = (kernel.width - 1) / 2;
    int kernel_height_offset = (kernel.height - 1) / 2;

    // check if the proceeding loop will be out of range for the input kernel
    if (std::abs(static_cast<int>(start_x)) < kernel_width_offset or
        std::abs(static_cast<int>(start_x - width)) < kernel_width_offset or
        std::abs(static_cast<int>(start_y)) < kernel_height_offset or
        std::abs(static_cast<int>(start_y - height)) < kernel_height_offset) {
        throw std::runtime_error("cannot compute dot product at this position - index would be out of range in convolution");
    }

    // iterate through all channels and kernel positions
    for (size_t d = 0; d < depth; d++) {
        for (size_t kh = 0; kh < kernel.height; kh++) {
            for (size_t kw = 0; kw < kernel.width; kw++) {
                sum +=
                    (*this)(d, start_y + kh - kernel_height_offset, start_x + kw - kernel_width_offset) * kernel(d, kh, kw);
            }
        }
    }
    return sum;
}

// return a new tensor with the width and height axis padded by 'amount'.
Tensor3D Tensor3D::pad(const Tensor3D &input, int amount) {
    Tensor3D output(input.depth, input.height + 2 * amount, input.width + 2 * amount);
    for (int depth_index = 0; depth_index < output.depth; ++depth_index) {
        for (int height_index = amount; height_index < output.height - amount; ++height_index) {
            for (int width_index = amount; width_index < output.width - amount; ++width_index) {
                output(depth_index, height_index, width_index) =
                    input(depth_index, height_index - amount, width_index - amount);
            }
        }
    }
    return output;
}

// initialization methods
void Tensor3D::he_initialise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / (height * width * depth));
    std::normal_distribution<float> dis(0.0f, std_dev);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3D::xavier_initialise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (height * width * depth));
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3D::uniform_initialise(float lower_bound, float upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower_bound, upper_bound);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3D::zero_initialise() {
    for (auto &element : data) {
        element = 0.0f;
    }
}

// operators

/**
 * @brief Overloads the multiplication operator for Tensor3D multiplication on each depth slice.
 * @param other The tensor to multiply with.
 * @return The resulting tensor after multiplication.
 */
Tensor3D Tensor3D::operator*(const Tensor3D &other) const {
    // check dimensions match for Tensor3D multiplication at each depth
    if (width != other.height) {
        throw std::invalid_argument("tensor dimensions don't match for multiplication: (" + std::to_string(height) + "x" +
                                    std::to_string(width) + "x" + std::to_string(depth) + ") * (" +
                                    std::to_string(other.height) + "x" + std::to_string(other.width) + "x" +
                                    std::to_string(other.depth) + ")");
    }
    if (depth != other.depth) {
        throw std::invalid_argument("tensor depths must match for multiplication");
    }

    // result will have dimensions: (this.height x other.width x depth)
    Tensor3D result(depth, height, other.width);

    // perform Tensor3D multiplication for each depth slice
    for (size_t d = 0; d < depth; d++) {
        // cache-friendly loop order (k before j)
        for (size_t i = 0; i < height; i++) {
            for (size_t k = 0; k < width; k++) {
                for (size_t j = 0; j < other.width; j++) {
                    result(d, i, j) += (*this)(d, i, k) * other(d, k, j);
                }
            }
        }
    }

    return result;
}

Tensor3D Tensor3D::operator+(const Tensor3D &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for addition");
    }

    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) + other(d, i, j);
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::operator+(const float &other) const {
    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) + other;
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::operator-(const Tensor3D &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for subtraction");
    }

    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) - other(d, i, j);
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::operator*(float scalar) const {
    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) * scalar;
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::hadamard(const Tensor3D &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for Hadamard product");
    }

    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) * other(d, i, j);
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::apply(float (*func)(float)) const {
    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = func((*this)(d, i, j));
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::transpose() const {
    Tensor3D result(depth, width, height);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, j, i) = (*this)(d, i, j);
            }
        }
    }
    return result;
}

// softmax across height dimension for back-compatibility with old Tensor3D class
Tensor3D Tensor3D::softmax() const {
    Tensor3D result(depth, height, width);

    for (size_t d = 0; d < depth; d++) {
        // find max across height (class scores)
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t h = 0; h < height; h++) {
            max_val = std::max(max_val, (*this)(d, h, 0));
        }

        // compute exp and sum across height
        float sum = 0.0f;
        for (size_t h = 0; h < height; h++) {
            result(d, h, 0) = std::exp((*this)(d, h, 0) - max_val);
            sum += result(d, h, 0);
        }

        // normalize across height
        for (size_t h = 0; h < height; h++) {
            result(d, h, 0) /= sum;
        }
    }
    return result;
}

Tensor3D Tensor3D::flatten() const {
    // create tensor of shape (1, depth*height*width, 1)
    Tensor3D result(1, depth * height * width, 1);

    // copy values sequentially
    size_t idx = 0;
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                result(0, idx, 0) = (*this)(d, h, w);
                idx++;
            }
        }
    }

    return result;
}

Tensor3D Tensor3D::unflatten(size_t new_depth, size_t new_height, size_t new_width) const {
    // check if dimensions match
    if (depth != 1 or width != 1 or height != new_depth * new_height * new_width) {
        throw std::runtime_error("cannot unflatten tensor - dimensions don't match. Expected flattened tensor of height " +
                                    std::to_string(new_depth * new_height * new_width) + " but got height " +
                                    std::to_string(height));
    }

    Tensor3D result(new_depth, new_height, new_width);
    size_t idx = 0;

    // copy values back to 3D structure
    for (size_t d = 0; d < new_depth; d++) {
        for (size_t h = 0; h < new_height; h++) {
            for (size_t w = 0; w < new_width; w++) {
                result(d, h, w) = (*this)(0, idx, 0);
                idx++;
            }
        }
    }

    return result;
}

Tensor3D Tensor3D::Conv(const Tensor3D &input, const Tensor3D &kernel) {
    // check dimensions
    if (input.depth != kernel.depth) {
        throw std::runtime_error("input and kernel must have same depth for convolution");
    }

    // perform full convolution (no padding)
    Tensor3D output(1, input.height - kernel.height + 1, input.width - kernel.width + 1);

    // for each position in the output
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            float sum = 0.0f;

            // sum over all channels and kernel positions
            for (int d = 0; d < input.depth; ++d) {
                for (int ky = 0; ky < kernel.height; ++ky) {
                    for (int kx = 0; kx < kernel.width; ++kx) {
                        sum += input(d, y + ky, x + kx) * kernel(d, ky, kx);
                    }
                }
            }
            output(0, y, x) = sum;
        }
    }
    return output;
}

Tensor3D Tensor3D::rotate_180() const {
    Tensor3D result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                result(d, height - 1 - h, width - 1 - w) = (*this)(d, h, w);
            }
        }
    }
    return result;
}

void Tensor3D::set_depth_slice(size_t depth_index, const Tensor3D &slice) {
    if (depth_index >= depth) {
        throw std::runtime_error("depth_index out of range in set_depth_slice");
    }
    if (slice.depth != 1 or slice.height != height or slice.width != width) {
        throw std::runtime_error("slice dimensions don't match in set_depth_slice");
    }

    // Copy the entire slice
    std::copy(slice.data.begin(), slice.data.begin() + height * width, data.begin() + depth_index * height * width);
}

std::ostream &operator<<(std::ostream &os, const Tensor3D &tensor) {
    os << "Tensor3D(" << tensor.depth << ", " << tensor.height << ", " << tensor.width << ")\n";

    for (size_t d = 0; d < tensor.depth; ++d) {
        os << "Depth " << d << ":\n";
        for (size_t h = 0; h < tensor.height; ++h) {
            os << "[";
            for (size_t w = 0; w < tensor.width; ++w) {
                os << std::fixed << std::setprecision(4) << tensor(d, h, w);
                if (w < tensor.width - 1) os << ", ";
            }
            os << "]\n";
        }
        if (d < tensor.depth - 1) os << "\n";
    }
    return os;
}

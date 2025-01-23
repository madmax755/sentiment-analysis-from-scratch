#include "../include/Tensor3d.hpp"

#include <fstream>
#include <execinfo.h>  // for backtrace
#include <cxxabi.h>    // for demangling C++ names
#include <dlfcn.h>   // for dladdr
#include <cxxabi.h>  // for demangling
#include <stdio.h>
#include <assert.h>


Tensor3d::Tensor3d() : height(0), width(0), depth(0) {}

// legacy constructor for 2D matrix compatibility 
Tensor3d::Tensor3d(size_t rows, size_t cols) : depth(1), height(rows), width(cols) { 
    data.resize(depth * height * width); 
}

// construct from dimensions
Tensor3d::Tensor3d(size_t depth, size_t height, size_t width) : height(height), width(width), depth(depth) {
    data.resize(depth * height * width);
}

// construct from flat vector
Tensor3d::Tensor3d(size_t depth, size_t height, size_t width, const std::vector<float> &data)
    : height(height), width(width), depth(depth) {
    if (data.size() != depth * height * width) {
        throw std::invalid_argument("data length and dimension mismatch");
    }
    this->data = data;
}

// construct from 3D vector
Tensor3d::Tensor3d(size_t depth, size_t height, size_t width, const std::vector<std::vector<std::vector<float>>> &data)
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
size_t Tensor3d::index(size_t d, size_t h, size_t w) { return d * (height * width) + h * (width) + w; }
const size_t Tensor3d::index(size_t d, size_t h, size_t w) const { return d * (height * width) + h * (width) + w; }

// access elements using 3D coordinates
float &Tensor3d::operator()(size_t d, size_t h, size_t w) {
    if (d >= depth or h >= height or w >= width) {
        throw std::runtime_error("index out of range in Tensor3d::operator()");
    }
    return data[index(d, h, w)];
}
const float &Tensor3d::operator()(size_t d, size_t h, size_t w) const {
    if (d >= depth or h >= height or w >= width) {
        throw std::runtime_error("index out of range in Tensor3d::operator()");
    }
    return data[index(d, h, w)];
}

// extract 2D slice at given depth
Tensor3d Tensor3d::operator()(size_t d) {
    if (d >= depth) {
        throw std::runtime_error("index out of range in Tensor3d::operator()");
    }
    size_t slice_size = width * height;
    std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
    return Tensor3d(1, height, width, new_data);
}

const Tensor3d Tensor3d::operator()(size_t d) const {
    if (d >= depth) {
        throw std::runtime_error("index out of range in Tensor3d::operator()");
    }
    size_t slice_size = width * height;
    std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
    return Tensor3d(1, height, width, new_data);
}

// extract column at given columnindex
Tensor3d Tensor3d::col(int index) const {
    // convert negative index to positive
    if (index < 0) {index = width + index;}

    // check if index is within range
    if (index < 0 || index >= width) {throw std::runtime_error("index out of range in Tensor3d::col");}

    Tensor3d result(1, height, 1);

    for (size_t h = 0; h < height; h++) {
        result(0, h, 0) = (*this)(0, h, index);
    }

    return result;
}

std::vector<float>& Tensor3d::get_flat_data() { return data; }

const std::vector<float>& Tensor3d::get_flat_data() const { return data; }

// compute dot product with a kernel centered at specific position - the argument must be the kernel
float Tensor3d::dot_with_kernel_at_position(const Tensor3d &kernel, size_t start_x, size_t start_y) const {
    if (kernel.depth != depth) {
        throw std::runtime_error("kernel depth must match input tensor depth");
    }

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
Tensor3d Tensor3d::pad(const Tensor3d &input, int amount) {
    Tensor3d output(input.depth, input.height + 2 * amount, input.width + 2 * amount);
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
void Tensor3d::he_initialise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / (height * width * depth));
    std::normal_distribution<float> dis(0.0f, std_dev);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3d::xavier_initialise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (height * width * depth));
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3d::uniform_initialise(float lower_bound, float upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower_bound, upper_bound);

    for (auto &element : data) {
        element = dis(gen);
    }
}

void Tensor3d::zero_initialise() {
    for (auto &element : data) {
        element = 0.0f;
    }
}

// operators

/**
 * @brief Overloads the multiplication operator for Tensor3d multiplication on each depth slice.
 * @param other The tensor to multiply with.
 * @return The resulting tensor after multiplication.
 */
Tensor3d Tensor3d::operator*(const Tensor3d &other) const {
    // check dimensions match for Tensor3d multiplication at each depth
    if (width != other.height) {
        throw std::invalid_argument("tensor dimensions don't match for multiplication: (" + std::to_string(depth) + "x" +
                                    std::to_string(height) + "x" + std::to_string(width) + ") * (" +
                                    std::to_string(other.depth) + "x" + std::to_string(other.height) + "x" +
                                    std::to_string(other.width) + ")");
    }
    if (depth != other.depth) {
        throw std::invalid_argument("tensor depths must match for multiplication");
    }

    // result will have dimensions: (this.height x other.width x depth)
    Tensor3d result(depth, height, other.width);

    // perform Tensor3d multiplication for each depth slice
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

    result.check_for_nans("multiplication");
    return result;
}

Tensor3d Tensor3d::operator+(const Tensor3d &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for addition");
    }

    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) + other(d, i, j);
            }
        }
    }
    result.check_for_nans("addition");
    return result;
}

Tensor3d Tensor3d::operator+(const float &other) const {
    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) + other;
            }
        }
    }
    result.check_for_nans("scalar addition");
    return result;
}

Tensor3d Tensor3d::operator-(const Tensor3d &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for subtraction");
    }

    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) - other(d, i, j);
            }
        }
    }
    result.check_for_nans("subtraction");
    return result;
}

Tensor3d Tensor3d::operator*(float scalar) const {
    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) * scalar;
            }
        }
    }
    result.check_for_nans("scalar multiplication");
    return result;
}

Tensor3d Tensor3d::hadamard(const Tensor3d &other) const {
    if (height != other.height or width != other.width or depth != other.depth) {
        throw std::invalid_argument("tensor dimensions don't match for Hadamard product");
    }

    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = (*this)(d, i, j) * other(d, i, j);
            }
        }
    }
    result.check_for_nans("hadamard product");
    return result;
}

Tensor3d Tensor3d::apply(float (*func)(float)) const {
    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, i, j) = func((*this)(d, i, j));
            }
        }
    }
    result.check_for_nans("apply function");
    return result;
}

Tensor3d Tensor3d::transpose() const {
    Tensor3d result(depth, width, height);
    for (size_t d = 0; d < depth; d++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result(d, j, i) = (*this)(d, i, j);
            }
        }
    }
    result.check_for_nans("transpose");
    return result;
}

Tensor3d Tensor3d::softmax(std::string dim) const {
    if (depth != 1) {
        throw std::runtime_error("softmax_along_dim only supports 1xnxm tensors");
    }
    if (dim != "height" and dim != "width") {
        throw std::runtime_error("dimension must be height or width");
    }

    Tensor3d result(1, height, width);
    
    if (dim == "height") {  // softmax across height dimension
        // for each column
        for (size_t w = 0; w < width; w++) {
            // find max value in this column
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t h = 0; h < height; h++) {
                max_val = std::max(max_val, (*this)(0, h, w));
            }

            // compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t h = 0; h < height; h++) {
                result(0, h, w) = std::exp((*this)(0, h, w) - max_val);
                sum += result(0, h, w);
            }

            // normalise
            for (size_t h = 0; h < height; h++) {
                result(0, h, w) /= sum;
            }
        }
    } else {  // softmax across width dimension
        // for each row
        for (size_t h = 0; h < height; h++) {
            // find max value in this row
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t w = 0; w < width; w++) {
                max_val = std::max(max_val, (*this)(0, h, w));
            }

            // compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t w = 0; w < width; w++) {
                result(0, h, w) = std::exp((*this)(0, h, w) - max_val);
                sum += result(0, h, w);
            }

            // normalise
            for (size_t w = 0; w < width; w++) {
                result(0, h, w) /= sum;
            }
        }
    }

    result.check_for_nans("softmax");
    return result;
}

Tensor3d Tensor3d::flatten() const {
    // create tensor of shape (1, depth*height*width, 1)
    Tensor3d result(1, depth * height * width, 1);

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

    result.check_for_nans("flatten");
    return result;
}

Tensor3d Tensor3d::unflatten(size_t new_depth, size_t new_height, size_t new_width) const {
    // check if dimensions match
    if (depth != 1 or width != 1 or height != new_depth * new_height * new_width) {
        throw std::runtime_error("cannot unflatten tensor - dimensions don't match. Expected flattened tensor of height " +
                                    std::to_string(new_depth * new_height * new_width) + " but got height " +
                                    std::to_string(height));
    }

    Tensor3d result(new_depth, new_height, new_width);
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

    result.check_for_nans("unflatten");
    return result;
}

Tensor3d Tensor3d::Conv(const Tensor3d &input, const Tensor3d &kernel) {
    // check dimensions
    if (input.depth != kernel.depth) {
        throw std::runtime_error("input and kernel must have same depth for convolution");
    }

    // perform full convolution (no padding)
    Tensor3d output(1, input.height - kernel.height + 1, input.width - kernel.width + 1);

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
    output.check_for_nans("convolution");
    return output;
}

Tensor3d Tensor3d::rotate_180() const {
    Tensor3d result(depth, height, width);
    for (size_t d = 0; d < depth; d++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                result(d, height - 1 - h, width - 1 - w) = (*this)(d, h, w);
            }
        }
    }
    result.check_for_nans("rotate_180");
    return result;
}

// returns a diagonal matrix from a vector
Tensor3d Tensor3d::diag() const {
    // check if matrix is of form 1xnx1 or 1x1xn
    if (depth != 1 or (height != 1 and width != 1)) {
        throw std::runtime_error("matrix is not of form 1xnx1 or 1x1xn for diag");
    }

    // if height == 1, then the matrix is of form 1xnx1
    // if width == 1, then the matrix is of form 1x1xn

    if (height == 1) {
        Tensor3d result(1, width, width);
        for (size_t i = 0; i < width; i++) {
            result(0, i, i) = (*this)(0, 0, i);
        }
        return result;
    } else {
        Tensor3d result(1, height, height);
        for (size_t i = 0; i < height; i++) {
            result(0, i, i) = (*this)(0, i, 0);
        }
        return result;
    }
}

void Tensor3d::set_depth_slice(size_t depth_index, const Tensor3d &slice) {
    if (depth_index >= depth) {
        throw std::runtime_error("depth_index out of range in set_depth_slice");
    }
    if (slice.depth != 1 or slice.height != height or slice.width != width) {
        throw std::runtime_error("slice dimensions don't match in set_depth_slice");
    }

    // Copy the entire slice
    std::copy(slice.data.begin(), slice.data.begin() + height * width, data.begin() + depth_index * height * width);
}

std::ostream &operator<<(std::ostream &os, const Tensor3d &tensor) {
    os << "Tensor3d(" << tensor.depth << ", " << tensor.height << ", " << tensor.width << ")\n";

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

// helper functions for saving/loading Tensor3d
void Tensor3d::save_to_file(std::ofstream &file) const {
    // write dimensions
    uint32_t depth_val = static_cast<uint32_t>(depth);
    uint32_t height_val = static_cast<uint32_t>(height); 
    uint32_t width_val = static_cast<uint32_t>(width);
    
    file.write(reinterpret_cast<const char *>(&depth_val), sizeof(depth_val));
    file.write(reinterpret_cast<const char *>(&height_val), sizeof(height_val));
    file.write(reinterpret_cast<const char *>(&width_val), sizeof(width_val));

    // write flattened data
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

void Tensor3d::load_from_file(std::ifstream &file) {
    // read dimensions
    uint32_t depth_val, height_val, width_val;
    file.read(reinterpret_cast<char *>(&depth_val), sizeof(depth_val));
    file.read(reinterpret_cast<char *>(&height_val), sizeof(height_val));
    file.read(reinterpret_cast<char *>(&width_val), sizeof(width_val));

    depth = static_cast<size_t>(depth_val);
    height = static_cast<size_t>(height_val);
    width = static_cast<size_t>(width_val);

    // resize data vector before reading
    data.resize(depth * height * width);

    // read flattened data
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
}


std::pair<float, float> Tensor3d::get_magnitudes() const {
    float max_val = 0.0f;
    float sum = 0.0f;
    int count = data.size();
    
    for (const float& val : data) {
        float abs_val = std::abs(val);
        max_val = std::max(max_val, abs_val);
        sum += abs_val;
    }
    
    return {max_val, sum / count};
}


//debug 
void Tensor3d::check_for_nans(const std::string& operation) const {
    for (const auto& val : data) {
        if (std::isnan(val) || std::isinf(val)) {
            if (std::isnan(val)) {
                std::cerr << "NaN detected in Tensor3d during: " << operation << "\n";
            } else {
                std::cerr << "Inf detected in Tensor3d during: " << operation << "\n";
            }
            
            // get backtrace
            void* callstack[128];
            int frames = backtrace(callstack, 128);
            char** strs = backtrace_symbols(callstack, frames);
            
            // print stack trace with demangled names
            for (int i = 0; i < frames; ++i) {
                Dl_info info;
                if (dladdr(callstack[i], &info)) {
                    // demangle the C++ name
                    int status;
                    char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);
                    const char* name = status == 0 ? demangled : info.dli_sname;
                    
                    std::cerr << "Frame " << i << ": " 
                             << (name ? name : "??") 
                             << " in " << info.dli_fname 
                             << " at " << callstack[i] << "\n";
                             
                    free(demangled);
                } else {
                    // fallback to original backtrace string if dladdr fails
                    std::cerr << strs[i] << "\n";
                }
            }
            free(strs);
            
            throw std::runtime_error("NaN detected");
        }
    }
}

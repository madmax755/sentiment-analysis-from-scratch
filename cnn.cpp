#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------- ACTIVATION FUNCTIONS -------------------------------------------

// sigmoid activation function
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// sigmoid derivative
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// relu activation function
float relu(float x) { return std::max(x, 0.0f); }

// relu derivative
float relu_derivative(float x) { return (x > 0.0f) ? 1.0f : 0.0f; }

// read binary file into a vector
std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (file) {
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
        return buffer;
    } else {
        std::cout << "Error reading file " << path << "\n";
        return std::vector<unsigned char>();  // return an empty vector
    }
}

class Tensor3D {
   private:
    // store as 1D vector for cache efficiency
    // access pattern: data[d*height*width + h*width + w]
    std::vector<float> data;

   public:
    size_t height, width, depth;

    Tensor3D() : height(0), width(0), depth(0) {}

    // legacy constructor for 2D matrix compatibility 
    Tensor3D(size_t rows, size_t cols) : depth(1), height(rows), width(cols) { 
        data.resize(depth * height * width); 
    }

    // construct from dimensions
    Tensor3D(size_t depth, size_t height, size_t width) : height(height), width(width), depth(depth) {
        data.resize(depth * height * width);
    }

    // construct from flat vector
    Tensor3D(size_t depth, size_t height, size_t width, const std::vector<float> &data)
        : height(height), width(width), depth(depth) {
        if (data.size() != depth * height * width) {
            throw std::invalid_argument("data length and dimension mismatch");
        }
        this->data = data;
    }

    // construct from 3D vector
    Tensor3D(size_t depth, size_t height, size_t width, const std::vector<std::vector<std::vector<float>>> &data)
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
    size_t index(size_t d, size_t h, size_t w) { return d * (height * width) + h * (width) + w; }
    const size_t index(size_t d, size_t h, size_t w) const { return d * (height * width) + h * (width) + w; }

    // access elements using 3D coordinates
    float &operator()(size_t d, size_t h, size_t w) { return data[index(d, h, w)]; }
    const float &operator()(size_t d, size_t h, size_t w) const { return data[index(d, h, w)]; }

    // extract 2D slice at given depth
    Tensor3D operator()(size_t d) {
        size_t slice_size = width * height;
        std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
        return Tensor3D(1, height, width, new_data);
    }

    const Tensor3D operator()(size_t d) const {
        size_t slice_size = width * height;
        std::vector<float> new_data(data.begin() + d * slice_size, data.begin() + d * slice_size + slice_size);
        return Tensor3D(1, height, width, new_data);
    }

    std::vector<float>& get_flat_data() { return data; }

    const std::vector<float>& get_flat_data() const { return data; }

    // compute dot product with a kernel centered at specific position - the argument must be the kernel
    float dot_with_kernel_at_position(const Tensor3D &kernel, size_t start_x, size_t start_y) const {
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
    static Tensor3D pad(const Tensor3D &input, int amount = 1) {
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
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float std_dev = std::sqrt(2.0f / (height * width * depth));
        std::normal_distribution<float> dis(0.0f, std_dev);

        for (auto &element : data) {
            element = dis(gen);
        }
    }

    void xavier_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = std::sqrt(6.0f / (height * width * depth));
        std::uniform_real_distribution<float> dis(-limit, limit);

        for (auto &element : data) {
            element = dis(gen);
        }
    }

    void uniform_initialise(float lower_bound = 0.0f, float upper_bound = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(lower_bound, upper_bound);

        for (auto &element : data) {
            element = dis(gen);
        }
    }

    void zero_initialise() {
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
    Tensor3D operator*(const Tensor3D &other) const {
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

    Tensor3D operator+(const Tensor3D &other) const {
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

    Tensor3D operator+(const float &other) const {
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

    Tensor3D operator-(const Tensor3D &other) const {
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

    Tensor3D operator*(float scalar) const {
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

    Tensor3D hadamard(const Tensor3D &other) const {
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

    Tensor3D apply(float (*func)(float)) const {
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

    template <typename Func>
    Tensor3D apply(Func func) const {
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

    Tensor3D transpose() const {
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
    Tensor3D softmax() const {
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

    Tensor3D flatten() const {
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

    Tensor3D unflatten(size_t new_depth, size_t new_height, size_t new_width) const {
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

    static Tensor3D Conv(const Tensor3D &input, const Tensor3D &kernel) {
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

    Tensor3D rotate_180() const {
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

    void set_depth_slice(size_t depth_index, const Tensor3D &slice) {
        if (depth_index >= depth) {
            throw std::runtime_error("depth_index out of range in set_depth_slice");
        }
        if (slice.depth != 1 or slice.height != height or slice.width != width) {
            throw std::runtime_error("slice dimensions don't match in set_depth_slice");
        }

        // Copy the entire slice
        std::copy(slice.data.begin(), slice.data.begin() + height * width, data.begin() + depth_index * height * width);
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor3D &tensor) {
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
};

struct BackwardReturn {
    Tensor3D input_error;
    std::vector<Tensor3D> weight_grads;
    std::vector<Tensor3D> bias_grads;
};

// ---------------------------------- LAYER CLASSES -------------------------------------------
class Layer {
   public:
    std::vector<Tensor3D> weights;
    std::vector<Tensor3D> biases;
    virtual ~Layer() = default;

    // pure virtual class - requires implementation in derived objects.
    virtual Tensor3D forward(const Tensor3D &input) = 0;
    virtual BackwardReturn backward(const Tensor3D &d_output) = 0;
};

class DenseLayer : public Layer {
   public:
    std::string activation_function;
    Tensor3D input;
    Tensor3D z;

    /**
     * @brief Constructs a DenseLayer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    DenseLayer(size_t input_size, size_t output_size, std::string activation_function = "relu")
        : activation_function(activation_function) {
        weights.emplace_back(output_size, input_size);
        biases.emplace_back(output_size, 1);

        if (activation_function == "sigmoid") {
            weights[0].xavier_initialise();
        } else if (activation_function == "relu") {
            weights[0].he_initialise();
        } else if (activation_function == "softmax") {
            weights[0].uniform_initialise(0, 1);
        } else {
            weights[0].uniform_initialise();
        }
    }

    // returns post-activation - stores input and preactivation for backprop
    Tensor3D forward(const Tensor3D &input) override {
        if (weights[0].width != input.height) {
            throw std::runtime_error("Input vector dimension is not appropriate for the weight Tensor3D dimension");
        }

        this->input = input;  // store for backprop
        z = weights[0] * input + biases[0];  // compute pre-activation

        Tensor3D a;
        // apply appropriate activation function
        if (activation_function == "sigmoid") {
            a = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            a = z.apply(relu);
        } else if (activation_function == "softmax") {
            // softmax special case - apply across height dimension
            a = z.softmax(); 
        } else {
            a = z;  // no activation ("none")
        }

        return a;
    }

    BackwardReturn backward(const Tensor3D &d_output) override {
        Tensor3D d_activation;
        Tensor3D d_z;

        // compute gradient based on activation function
        if (activation_function == "sigmoid") {
            d_activation = z.apply(sigmoid_derivative);
            d_z = d_output.hadamard(d_activation);
        } else if (activation_function == "relu") {
            d_activation = z.apply(relu_derivative);
            d_z = d_output.hadamard(d_activation);
        } else if (activation_function == "softmax" or activation_function == "none") {
            // for softmax with cross-entropy loss, gradient simplifies to (predicted - target)
            d_z = d_output;
        } else {
            throw std::runtime_error("Unsupported activation function");
        }

        // compute gradients using chain rule
        Tensor3D d_input = weights[0].transpose() * d_z;  // gradient w.r.t input
        std::vector<Tensor3D> d_weights = {d_z * input.transpose()};  // gradient w.r.t weights
        std::vector<Tensor3D> d_biases = {d_z};  // gradient w.r.t biases

        return {d_input, d_weights, d_biases};
    }
};

class ConvolutionLayer : public Layer {
   public:
    int channels_in;      // number of input channels/feature maps
    int out_channels;     // number of output feature maps to produce
    int kernel_size;      // size of convolutional kernel (assumed square)
    std::string mode;     // padding mode ('same' maintains input dimensions)

    Tensor3D input;       // store input for backward pass
    Tensor3D z;          // store pre-activations for backward pass

    ConvolutionLayer(int channels_in, int out_channels, int kernel_size, std::string mode = "same")
        : channels_in(channels_in), out_channels(out_channels), kernel_size(kernel_size), mode(mode) {
        // create kernel tensors - one per output feature map
        weights.reserve(out_channels);
        for (int i = 0; i < out_channels; i++) {
            weights.emplace_back(channels_in, kernel_size, kernel_size);
            weights.back().he_initialise();  // he initialisation for ReLU activation
        }

        // initialise biases as 1x1x1 tensors with zeros
        biases.resize(out_channels, Tensor3D(1, 1, 1));
    }

    Tensor3D forward(const Tensor3D &input) override {
        // validate input dimensions match layer configuration
        if (input.depth != channels_in) {
            throw std::runtime_error("Input tensor depth does not match the number of input channels for this layer");
        }
        if (input.height == 0 or input.width == 0) {
            throw std::runtime_error("Input tensor has zero height or width");
        }

        this->input = input;  // store for backward pass
        // output will have same spatial dimensions but depth = number of filters
        Tensor3D a(weights.size(), input.height, input.width);
        Tensor3D tmp_z(weights.size(), input.height, input.width);

        if (mode == "same") {
            // pad input with zeros to maintain spatial dimensions after convolution
            // padding = (kernel_size - 1)/2 on all sides
            Tensor3D padded_input = Tensor3D::pad(input, (kernel_size - 1) / 2);

            // compute each output feature map
            for (int feature_map_index = 0; feature_map_index < weights.size(); ++feature_map_index) {
                // convolve input with kernel and add bias
                // Conv() performs: Σ(input channel * kernel) over all channels
                Tensor3D preactivation = 
                    Tensor3D::Conv(padded_input, weights[feature_map_index]) + biases[feature_map_index](0, 0, 0);

                // store pre-activation for backprop
                tmp_z.set_depth_slice(feature_map_index, preactivation);

                // apply ReLU and store in output tensor
                a.set_depth_slice(feature_map_index, preactivation.apply(relu));
            }

            z = tmp_z;  // store all pre-activations for backprop
            return a;
        }
        else {
            throw std::runtime_error("Convolution layer mode not specified or handled correctly");
        }
    }

    BackwardReturn backward(const Tensor3D &d_output) override {
        // element-wise product of output gradient with ReLU derivative
        Tensor3D d_z = d_output.hadamard(z.apply(relu_derivative));

        std::vector<Tensor3D> d_weights;
        std::vector<Tensor3D> d_bias;
        d_weights.reserve(weights.size());
        d_bias.reserve(weights.size());

        // will accumulate gradients w.r.t input
        Tensor3D d_input(input.depth, input.height, input.width);
        d_input.zero_initialise();

        int pad_amount = (kernel_size - 1) / 2;
        // pad gradient tensor for full convolution
        Tensor3D padded_d_z = Tensor3D::pad(d_z, pad_amount);

        // compute input gradients using convolution with rotated kernels
        for (int in_ch = 0; in_ch < input.depth; in_ch++) {
            for (int k = 0; k < weights.size(); k++) {
                Tensor3D relevant_d_z_slice = padded_d_z(k);
                // rotate kernel 180° - required for convolution gradient computation
                Tensor3D relevant_kernel_slice = weights[k](in_ch).rotate_180();

                // convolve gradient with rotated kernel
                Tensor3D d_input_contribution = Tensor3D::Conv(relevant_d_z_slice, relevant_kernel_slice);

                // accumulate contributions from each output channel
                for (int y = 0; y < d_input.height; y++) {
                    for (int x = 0; x < d_input.width; x++) {
                        d_input(in_ch, y, x) += d_input_contribution(0, y, x);
                    }
                }
            }
        }

        // compute weight gradients by convolving input with output gradients
        Tensor3D padded_input = Tensor3D::pad(input, pad_amount);

        for (int k = 0; k < weights.size(); k++) {
            // gradient tensor for k-th kernel
            Tensor3D d_weight(input.depth, kernel_size, kernel_size);
            
            for (int in_ch = 0; in_ch < input.depth; in_ch++) {
                // extract relevant slices for gradient computation
                Tensor3D padded_input_channel = padded_input(in_ch);
                Tensor3D d_z_channel = d_z(k);

                // compute gradient for this input-output channel pair
                Tensor3D channel_grad = Tensor3D::Conv(padded_input_channel, d_z_channel);
                d_weight.set_depth_slice(in_ch, channel_grad);
            }
            d_weights.push_back(d_weight);

            // bias gradient is sum of output gradients over spatial dimensions
            float d_bias_val = 0.0f;
            for (int y = 0; y < d_output.height; y++) {
                for (int x = 0; x < d_output.width; x++) {
                    d_bias_val += d_z(k, y, x);
                }
            }
            Tensor3D bias_grad(1, 1, 1);
            bias_grad(0, 0, 0) = d_bias_val;
            d_bias.push_back(bias_grad);
        }

        return {d_input, d_weights, d_bias};
    }
};

class PoolingLayer : public Layer {
   public:
    int kernel_size;      // size of pooling window (assumed square)
    int stride;           // step size between pooling operations
    std::string mode;     // pooling type (currently only "max" supported)
    Tensor3D input;       // store input for backward pass
    
    // stores (y,x) coordinates of max values for each pooling window
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_positions;  
    
    // cache output dimensions for handling flattened tensors
    size_t output_depth, output_height, output_width;

    PoolingLayer(int kernel_size = 2, int stride = -1, std::string mode = "max")
        : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride), mode(mode) {
        // stride should not exceed kernel size to avoid gaps
        if (stride > kernel_size) {
            throw std::runtime_error("stride should not be greater than kernel size");
        }
    }

    Tensor3D forward(const Tensor3D &input) override {
        this->input = input;  // store for backward pass
        
        // calculate output dimensions including partial window pooling
        output_height = std::ceil((input.height - kernel_size) / stride + 1);
        output_width = std::ceil((input.width - kernel_size) / stride + 1);
        output_depth = input.depth;
        
        Tensor3D output(output_depth, output_height, output_width);

        // initialise storage for max value positions
        max_positions.resize(input.depth);
        for (auto &channel : max_positions) {
            channel.resize(output_height, std::vector<std::pair<int, int>>(output_width));
        }

        if (mode == "max") {
            // process each channel independently
            for (int d = 0; d < input.depth; ++d) {
                for (int y = 0; y < output_height; ++y) {
                    for (int x = 0; x < output_width; ++x) {
                        // define boundaries of current pooling window
                        int start_y = y * stride;
                        int start_x = x * stride;
                        int end_y = std::min(start_y + kernel_size, static_cast<int>(input.height));
                        int end_x = std::min(start_x + kernel_size, static_cast<int>(input.width));

                        // find maximum value in current window
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_y = -1, max_x = -1;
                        
                        for (int wy = start_y; wy < end_y; ++wy) {
                            for (int wx = start_x; wx < end_x; ++wx) {
                                if (input(d, wy, wx) > max_val) {
                                    max_val = input(d, wy, wx);
                                    max_y = wy;
                                    max_x = wx;
                                }
                            }
                        }
                        
                        output(d, y, x) = max_val;
                        // store position of max value for backprop
                        max_positions[d][y][x] = {max_y, max_x};
                    }
                }
            }
        } else {
            throw std::runtime_error("mode not specified or handled correctly");
        }

        return output;
    }

    BackwardReturn backward(const Tensor3D &d_output) override {
        // handle case where gradient comes from dense layer
        Tensor3D d_output_unflattened = d_output;
        if (d_output.depth == 1 && d_output.width == 1) {
            // reshape gradient to match pooling layer output shape
            d_output_unflattened = d_output.unflatten(output_depth, output_height, output_width);
        }

        // initialise gradient tensor for input (same size as input)
        Tensor3D d_input(input.depth, input.height, input.width);
        d_input.zero_initialise();

        if (mode == "max") {
            // validate gradient dimensions match stored positions
            if (d_output_unflattened.depth != max_positions.size() or 
                d_output_unflattened.height != max_positions[0].size() or
                d_output_unflattened.width != max_positions[0][0].size()) {
                throw std::runtime_error(
                    "d_output dimensions do not match stored max_positions dimensions in pooling layer backward pass");
            }

            // propagate gradients only to positions where maximum was found
            for (size_t d = 0; d < d_output_unflattened.depth; ++d) {
                for (size_t y = 0; y < d_output_unflattened.height; ++y) {
                    for (size_t x = 0; x < d_output_unflattened.width; ++x) {
                        // get position where maximum was found during forward pass
                        auto [max_y, max_x] = max_positions[d][y][x];
                        // pass gradient only to that position
                        d_input(d, max_y, max_x) += d_output_unflattened(d, y, x);
                    }
                }
            }
        }

        // pooling layers have no learnable parameters
        std::vector<Tensor3D> empty_weight_grads;
        std::vector<Tensor3D> empty_bias_grads;

        return {d_input, empty_weight_grads, empty_bias_grads};
    }
};

// ---------------------------------- LOSS FUNCTIONS -------------------------------------------
// abstract base class for loss functions
class Loss {
   public:
    virtual ~Loss() = default;

    // compute scalar loss value between predicted and target tensors
    virtual float compute(const Tensor3D &predicted, const Tensor3D &target) const = 0;

    // compute gradient of loss with respect to network predictions
    virtual Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    float compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                // add small epsilon (1e-10) to prevent log(0)
                loss -= target(0, i, j) * std::log(predicted(0, i, j) + 1e-10f);
            }
        }
        return loss / predicted.width;  // average loss over batch
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        // when combined with softmax output, gradient simplifies to (predicted - target)
        // this is because d(cross_entropy)/d(softmax_input) = predicted - target
        return predicted - target;
    }
};

class MSELoss : public Loss {
   public:
    float compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        float loss = 0.0f;
        // sum squared differences between predictions and targets
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                float diff = predicted(0, i, j) - target(0, i, j);
                loss += diff * diff;
            }
        }
        // divide by 2 for easier derivative computation and average over batch
        return loss / (2.0f * predicted.width);
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        // derivative of MSE is (predicted - target) / batch_size
        return (predicted - target) * (1.0 / predicted.width);
    }
};

// ---------------------------------- OPTIMISERS -------------------------------------------
// abstract base class for parameter updates
class Optimiser {
   public:
    virtual ~Optimiser() = default;

    // update network parameters using computed gradients
    // layers: network layers containing weights/biases to update
    // gradients: pairs of (weight_grads, bias_grads) for each layer
    virtual void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) = 0;
};

// basic stochastic gradient descent optimiser
class SGDOptimiser : public Optimiser {
   private:
    float learning_rate;  // step size for parameter updates

   public:
    SGDOptimiser(float lr = 0.1f) : learning_rate(lr) {}

    void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) override {
        
        // iterate through each layer
        for (int layer_index = 0; layer_index < layers.size(); layer_index++) {
            auto [weight_gradients, bias_gradients] = gradients[layer_index];

            // update weights: w = w - lr * dw
            for (int weight_index = 0; weight_index < weight_gradients.size(); weight_index++) {
                (layers[layer_index])->weights[weight_index] =
                    (layers[layer_index])->weights[weight_index] - (weight_gradients[weight_index] * learning_rate);
            }

            // update biases: b = b - lr * db
            for (int bias_index = 0; bias_index < bias_gradients.size(); bias_index++) {
                (layers[layer_index])->biases[bias_index] =
                    (layers[layer_index])->biases[bias_index] - (bias_gradients[bias_index] * learning_rate);
            }
        }
    }
};

// AdamW optimiser: Adam with decoupled weight decay
class AdamWOptimiser : public Optimiser {
   private:
    float learning_rate;  // step size for parameter updates
    float beta1;         // exponential decay rate for first moment
    float beta2;         // exponential decay rate for second moment
    float epsilon;       // small value to prevent division by zero
    float weight_decay;  // L2 regularisation strength
    int t;              // timestep counter

    // momentum vectors for each parameter
    // first moment (mean) and second moment (uncentered variance)
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> m;  
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> v;  

   public:
    AdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    // initialise momentum vectors with zeros matching parameter shapes
    void initialize_moments(const std::vector<std::unique_ptr<Layer>> &layers) {
        m.clear();
        v.clear();

        for (const auto &layer : layers) {
            std::vector<Tensor3D> layer_weight_m, layer_weight_v;
            std::vector<Tensor3D> layer_bias_m, layer_bias_v;

            // initialise weight moments
            for (const auto &weight : layer->weights) {
                Tensor3D zero_tensor(weight.depth, weight.height, weight.width);
                layer_weight_m.push_back(zero_tensor);
                layer_weight_v.push_back(zero_tensor);
            }

            // initialise bias moments
            for (const auto &bias : layer->biases) {
                Tensor3D zero_tensor(bias.depth, bias.height, bias.width);
                layer_bias_m.push_back(zero_tensor);
                layer_bias_v.push_back(zero_tensor);
            }

            m.push_back({layer_weight_m, layer_bias_m});
            v.push_back({layer_weight_v, layer_bias_v});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(
        const std::vector<std::unique_ptr<Layer>> &layers,
        const std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> &gradients) override {
        
        // initialise momentum vectors if not already done
        if (m.empty() or v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep counter

        // update each layer's parameters
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            // update weights first
            for (int param_no = 0; param_no < m[layer_index].first.size(); ++param_no) {
                // update biased first moment estimate (momentum)
                m[layer_index].first[param_no] =
                    m[layer_index].first[param_no] * beta1 + gradients[layer_index].first[param_no] * (1.0 - beta1);

                // update biased second moment estimate (velocity)
                v[layer_index].first[param_no] =
                    v[layer_index].first[param_no] * beta2 +
                    gradients[layer_index].first[param_no].hadamard(gradients[layer_index].first[param_no]) * (1.0 - beta2);

                // compute bias-corrected moment estimates
                Tensor3D m_hat = m[layer_index].first[param_no] * (1.0 / (1.0 - std::pow(beta1, t)));
                Tensor3D v_hat = v[layer_index].first[param_no] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute Adam update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { 
                    return 1.0f / (std::sqrt(x) + epsilon); 
                }));

                // apply decoupled weight decay
                layers[layer_index]->weights[param_no] =
                    layers[layer_index]->weights[param_no] * (1.0 - learning_rate * weight_decay);
                
                // apply Adam update
                layers[layer_index]->weights[param_no] = 
                    layers[layer_index]->weights[param_no] - (update * learning_rate);
            }

            // update biases (same as weights but without weight decay)
            for (int param_no = 0; param_no < m[layer_index].second.size(); ++param_no) {
                m[layer_index].second[param_no] =
                    m[layer_index].second[param_no] * beta1 + gradients[layer_index].second[param_no] * (1.0 - beta1);

                v[layer_index].second[param_no] =
                    v[layer_index].second[param_no] * beta2 +
                    gradients[layer_index].second[param_no].hadamard(gradients[layer_index].second[param_no]) * (1.0 - beta2);

                Tensor3D m_hat = m[layer_index].second[param_no] * (1.0 / (1.0 - std::pow(beta1, t)));
                Tensor3D v_hat = v[layer_index].second[param_no] * (1.0 / (1.0 - std::pow(beta2, t)));

                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { 
                    return 1.0f / (std::sqrt(x) + epsilon); 
                }));

                // apply update (no weight decay for biases)
                layers[layer_index]->biases[param_no] = 
                    layers[layer_index]->biases[param_no] - (update * learning_rate);
            }
        }
    }
};

// ---------------------------------- NEURAL NETWORK -------------------------------------------
class NeuralNetwork {
   private:
    // specification for different layer types
    struct LayerSpec {
        enum Type { CONV, POOL, DENSE } type;

        // convolution parameters
        int in_channels = 0;
        int out_channels = 0;
        int kernel_size = 0;
        std::string mode = "same";

        // pooling parameters
        int pool_size = 0;
        int pool_stride = 0;
        std::string pool_mode = "max";

        // dense layer parameters
        std::string activation = "relu";
        size_t output_size = 0;

        // factory methods for creating layer specifications
        static LayerSpec Conv(int out_channels, int kernel_size, std::string mode = "same") {
            LayerSpec spec;
            spec.type = CONV;
            spec.out_channels = out_channels;
            spec.kernel_size = kernel_size;
            spec.mode = mode;
            return spec;
        }

        static LayerSpec Pool(int pool_size = 2, int stride = -1, std::string mode = "max") {
            LayerSpec spec;
            spec.type = POOL;
            spec.pool_size = pool_size;
            spec.pool_stride = (stride == -1) ? pool_size : stride;
            spec.pool_mode = mode;
            return spec;
        }

        static LayerSpec Dense(size_t output_size, std::string activation = "relu") {
            LayerSpec spec;
            spec.type = DENSE;
            spec.output_size = output_size;
            spec.activation = activation;
            return spec;
        }
    };

    // helper struct to track layer dimensions during network creation
    struct LayerDimensions {
        size_t height;
        size_t width;
        size_t depth;
    };

    void create_layers(const Tensor3D &input) {
        // track dimensions as we build the network
        LayerDimensions dims = {input.height, input.width, input.depth};

        for (auto &spec : layer_specs) {
            switch (spec.type) {
                case LayerSpec::CONV: {
                    spec.in_channels = dims.depth;
                    layers.push_back(
                        std::make_unique<ConvolutionLayer>(
                            spec.in_channels, spec.out_channels, 
                            spec.kernel_size, spec.mode
                        )
                    );
                    // only depth changes for 'same' padding
                    dims.depth = spec.out_channels;
                    break;
                }
                case LayerSpec::POOL: {
                    layers.push_back(
                        std::make_unique<PoolingLayer>(
                            spec.pool_size, spec.pool_stride, spec.pool_mode
                        )
                    );
                    // update dimensions after pooling
                    dims.height = std::floor((dims.height - spec.pool_size) / 
                        static_cast<float>(spec.pool_stride) + 1);
                    dims.width = std::floor((dims.width - spec.pool_size) / 
                        static_cast<float>(spec.pool_stride) + 1);
                    break;
                }
                case LayerSpec::DENSE: {
                    // flatten input for dense layer
                    int total_inputs = dims.height * dims.width * dims.depth;
                    layers.push_back(
                        std::make_unique<DenseLayer>(
                            total_inputs, spec.output_size, spec.activation
                        )
                    );
                    // output dimensions for dense layer
                    dims = {1, spec.output_size, 1};
                    break;
                }
            }
        }
        layers_created = true;
    }

    // helper functions for saving/loading Tensor3D
    static void save_tensor(std::ofstream &file, const Tensor3D &tensor) {
        // write dimensions
        uint32_t depth = static_cast<uint32_t>(tensor.depth);
        uint32_t height = static_cast<uint32_t>(tensor.height);
        uint32_t width = static_cast<uint32_t>(tensor.width);
        
        file.write(reinterpret_cast<const char *>(&depth), sizeof(depth));
        file.write(reinterpret_cast<const char *>(&height), sizeof(height));
        file.write(reinterpret_cast<const char *>(&width), sizeof(width));

        // write flattened data
        const auto &data = tensor.get_flat_data();
        file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    }

    static void load_tensor(std::ifstream &file, Tensor3D &tensor) {
        // read dimensions
        uint32_t depth, height, width;
        file.read(reinterpret_cast<char *>(&depth), sizeof(depth));
        file.read(reinterpret_cast<char *>(&height), sizeof(height));
        file.read(reinterpret_cast<char *>(&width), sizeof(width));

        // validate dimensions match expected tensor
        if (depth != tensor.depth || height != tensor.height || width != tensor.width) {
            throw std::runtime_error("tensor dimensions in file do not match expected dimensions");
        }

        // read flattened data
        auto &data = tensor.get_flat_data();
        file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    }

   public:
    struct EvalMetrics {
        float accuracy;
        float precision;
        float recall;
        float f1_score;

        friend std::ostream &operator<<(std::ostream &os, const EvalMetrics &metrics) {
            os << "accuracy: " << metrics.accuracy << ", precision: " << metrics.precision << ", recall: " << metrics.recall
               << ", f1_score: " << metrics.f1_score;
            return os;
        }
    };

    std::vector<LayerSpec> layer_specs;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimiser> optimiser;
    std::unique_ptr<Loss> loss;
    bool layers_created = false;

    // default constructor
    NeuralNetwork() {}

    // user-facing methods to add layers
    void add_conv_layer(int out_channels, int kernel_size, std::string mode = "same") {
        layer_specs.push_back(LayerSpec::Conv(out_channels, kernel_size, mode));
    }

    void add_pool_layer(int pool_size = 2, int stride = -1, std::string mode = "max") {
        layer_specs.push_back(LayerSpec::Pool(pool_size, stride, mode));
    }

    void add_dense_layer(int output_size, std::string activation = "relu") {
        layer_specs.push_back(LayerSpec::Dense(output_size, activation));
    }

    void set_optimiser(std::unique_ptr<Optimiser> new_optimiser) { 
        optimiser = std::move(new_optimiser); 
    }

    void set_loss(std::unique_ptr<Loss> new_loss) {
        // validate loss function compatibility with network architecture
        bool layers_empty = layer_specs.empty();
        bool last_layer_is_softmax = layer_specs.back().activation == "softmax";
        bool new_loss_is_cross_entropy = dynamic_cast<CrossEntropyLoss*>(new_loss.get()) != nullptr;

        if (layers_empty) {
            throw std::runtime_error("no layers created yet - set layers first");
        } else if (!last_layer_is_softmax and new_loss_is_cross_entropy) {
            throw std::runtime_error("last layer must be softmax for cross entropy loss");
        }

        loss = std::move(new_loss);
    }

    Tensor3D feedforward(const Tensor3D &input) {
        // create layers if this is first forward pass
        if (!layers_created) {
            create_layers(input);
        }

        Tensor3D current = input;
        
        // process through each layer
        for (size_t i = 0; i < layers.size(); i++) {
            // determine current and next layer types for transition handling
            auto *current_conv = dynamic_cast<ConvolutionLayer *>(layers[i].get());
            auto *current_dense = dynamic_cast<DenseLayer *>(layers[i].get());
            auto *current_pooling = dynamic_cast<PoolingLayer *>(layers[i].get());

            ConvolutionLayer *next_conv = nullptr;
            DenseLayer *next_dense = nullptr;
            PoolingLayer *next_pooling = nullptr;
            
            if (i + 1 < layers.size()) {
                next_conv = dynamic_cast<ConvolutionLayer *>(layers[i + 1].get());
                next_dense = dynamic_cast<DenseLayer *>(layers[i + 1].get());
                next_pooling = dynamic_cast<PoolingLayer *>(layers[i + 1].get());
            }

            // handle transitions between layer types
            if (current_conv) {
                current = current_conv->forward(current);
                // flatten if next layer is dense
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_pooling) {
                current = current_pooling->forward(current);
                // flatten if next layer is dense
                if (next_dense) {
                    current = current.flatten();
                }
            } else if (current_dense) {
                // dense layers can't be followed by conv/pool
                if (next_pooling or next_conv) {
                    throw std::runtime_error("dense layer cannot be followed by pooling or convolution");
                }
                current = current_dense->forward(current);
            } else {
                throw std::runtime_error("unknown layer type encountered");
            }
        }
        return current;
    }

    /**
     * @brief Performs backpropagation through the network for a single training example
     * @param input The input to the network
     * @param target The target output
     * @return A vector of pairs, each containing weight and bias gradients for a layer
     */
    std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> calculate_gradients(const Tensor3D &input,
                                                                                             const Tensor3D &target) {
        if (!loss) {
            throw std::runtime_error("loss function not set");
        }

        // forward pass to get predictions
        Tensor3D predicted = feedforward(input);

        // store gradients for each layer
        std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> all_gradients;
        all_gradients.reserve(layers.size());

        // initial gradient from loss function
        Tensor3D gradient = loss->derivative(predicted, target);

        // backpropagate through layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            BackwardReturn layer_grads = (*it)->backward(gradient);
            all_gradients.push_back({layer_grads.weight_grads, layer_grads.bias_grads});
            gradient = layer_grads.input_error;  // pass gradient to next layer
        }

        // reverse gradients to match layer order
        std::reverse(all_gradients.begin(), all_gradients.end());
        return all_gradients;
    }

    EvalMetrics evaluate(const std::vector<std::vector<Tensor3D>> &eval_set) {
        // track metrics for each class (0-9)
        int total_samples = 0;
        int correct_predictions = 0;
        std::vector<int> true_positives(10, 0);
        std::vector<int> false_positives(10, 0);
        std::vector<int> false_negatives(10, 0);

        // evaluate each sample
        for (const auto &sample : eval_set) {
            const auto &input = sample[0];
            const auto &target = sample[1];

            // get network prediction with augmentation
            Tensor3D predicted = feedforward(augment_image(input));

            // find predicted and actual class indices
            int predicted_class = 0;
            int actual_class = 0;
            float max_pred = predicted(0, 0, 0);
            float max_target = target(0, 0, 0);

            // find class with highest probability
            for (int i = 1; i < 10; ++i) {
                if (predicted(0, i, 0) > max_pred) {
                    max_pred = predicted(0, i, 0);
                    predicted_class = i;
                }
                if (target(0, i, 0) > max_target) {
                    max_target = target(0, i, 0);
                    actual_class = i;
                }
            }

            // update metrics
            total_samples++;
            if (predicted_class == actual_class) {
                correct_predictions++;
                true_positives[actual_class]++;
            } else {
                false_positives[predicted_class]++;
                false_negatives[actual_class]++;
            }
        }

        // calculate accuracy and per-class metrics
        float accuracy = static_cast<float>(correct_predictions) / total_samples;
        float total_precision = 0.0f;
        float total_recall = 0.0f;
        float total_f1 = 0.0f;
        int num_classes = 0;

        // compute macro-averaged metrics
        for (int i = 0; i < 10; ++i) {
            // skip classes with no samples
            if (true_positives[i] + false_positives[i] + false_negatives[i] == 0) {
                continue;
            }

            float class_precision = true_positives[i] / 
                static_cast<float>(true_positives[i] + false_positives[i]);
            float class_recall = true_positives[i] / 
                static_cast<float>(true_positives[i] + false_negatives[i]);
            float class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall);

            // handle division by zero cases
            if (std::isnan(class_f1)) {
                class_f1 = 0.0;
            }

            total_precision += class_precision;
            total_recall += class_recall;
            total_f1 += class_f1;
            num_classes++;
        }

        return {
            accuracy,
            total_precision / num_classes,  // macro-averaged precision
            total_recall / num_classes,     // macro-averaged recall
            total_f1 / num_classes         // macro-averaged F1
        };
    }

    void train(std::vector<std::vector<Tensor3D>> &training_set, 
              const std::vector<std::vector<Tensor3D>> &eval_set,
              const int num_epochs, const int batch_size) {
        
        const int batches_per_epoch = training_set.size() / batch_size;
        std::cout << "started training" << std::endl;

        // training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // shuffle training data at start of each epoch
            std::shuffle(training_set.begin(), training_set.end(), std::random_device());
            float epoch_loss = 0.0f;

            // process each batch
            for (int batch = 0; batch < batches_per_epoch; ++batch) {
                std::vector<std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>> batch_gradients;
                float batch_loss = 0.0f;

                // accumulate gradients over batch
                for (int i = 0; i < batch_size; ++i) {
                    int idx = batch * batch_size + i;
                    auto &input = training_set[idx][0];
                    auto &target = training_set[idx][1];

                    // apply data augmentation
                    auto augmented_input = augment_image(input);
                    auto gradients = calculate_gradients(augmented_input, target);

                    // initialise or accumulate batch gradients
                    if (i == 0) {
                        batch_gradients = gradients;
                    } else {
                        // add gradients element-wise across all layers
                        for (size_t layer = 0; layer < gradients.size(); ++layer) {
                            for (size_t w = 0; w < gradients[layer].first.size(); ++w) {
                                batch_gradients[layer].first[w] = 
                                    batch_gradients[layer].first[w] + gradients[layer].first[w];
                            }
                            for (size_t b = 0; b < gradients[layer].second.size(); ++b) {
                                batch_gradients[layer].second[b] = 
                                    batch_gradients[layer].second[b] + gradients[layer].second[b];
                            }
                        }
                    }

                    batch_loss += loss->compute(feedforward(input), target);
                }

                // average gradients and loss over batch
                for (auto &layer_grads : batch_gradients) {
                    for (auto &w_grad : layer_grads.first) {
                        w_grad = w_grad * (1.0f / batch_size);
                    }
                    for (auto &b_grad : layer_grads.second) {
                        b_grad = b_grad * (1.0f / batch_size);
                    }
                }
                batch_loss /= batch_size;
                epoch_loss += batch_loss;

                // update network parameters
                optimiser->compute_and_apply_updates(layers, batch_gradients);

                // evaluate and print progress every 30 batches
                if (batch % 30 == 0) {
                    EvalMetrics metrics = evaluate(eval_set);
                    std::cout << "epoch " << epoch + 1 << ", batch " << batch << "/" 
                             << batches_per_epoch << ": " << metrics << std::endl;
                }
            }

            // save model checkpoint after each epoch
            std::string model_path = "model" + std::to_string(epoch) + ".bin";
            save_model(model_path);
        }
    }

    // apply random transformations to input image for data augmentation
    Tensor3D augment_image(const Tensor3D &input, float offset_range = 5.0f, 
                          float scale_range = 0.2f, float angle_range = 20.0f) {
        
        Tensor3D augmented(1, 28, 28);
        std::mt19937 gen(std::random_device{}());

        // random translation (-5 to 5 pixels)
        std::uniform_int_distribution<> offset_dist(-offset_range, offset_range);
        int offsetX = offset_dist(gen);
        int offsetY = offset_dist(gen);

        // random rotation (-20 to 20 degrees)
        std::uniform_real_distribution<> angle_dist(-angle_range, angle_range);
        float angle = angle_dist(gen) * M_PI / 180.0f;

        // random scaling (0.8 to 1.2)
        std::uniform_real_distribution<> scale_dist(1.0f - scale_range, 1.0f + scale_range);
        float scale = scale_dist(gen);

        // center point for transformations
        float centerX = input.width / 2.0f;
        float centerY = input.height / 2.0f;

        // apply transformations using inverse mapping
        for (size_t y = 0; y < augmented.height; y++) {
            for (size_t x = 0; x < augmented.width; x++) {
                // translate to origin
                float dx = x - centerX - offsetX;
                float dy = y - centerY - offsetY;

                // apply inverse scale
                dx /= scale;
                dy /= scale;

                // apply inverse rotation
                float srcX = dx * cos(-angle) - dy * sin(-angle) + centerX;
                float srcY = dx * sin(-angle) + dy * cos(-angle) + centerY;

                // bilinear interpolation for smooth transformation
                if (srcX >= 0 && srcX < input.width - 1 && srcY >= 0 && srcY < input.height - 1) {
                    int x0 = static_cast<int>(srcX);
                    int x1 = x0 + 1;
                    int y0 = static_cast<int>(srcY);
                    int y1 = y0 + 1;

                    float wx1 = srcX - x0;
                    float wx0 = 1 - wx1;
                    float wy1 = srcY - y0;
                    float wy0 = 1 - wy1;

                    // interpolate between neighboring pixels
                    augmented(0, y, x) = 
                        input(0, y0, x0) * wx0 * wy0 + 
                        input(0, y0, x1) * wx1 * wy0 +
                        input(0, y1, x0) * wx0 * wy1 + 
                        input(0, y1, x1) * wx1 * wy1;
                }
            }
        }

        return augmented;
    }

    // save model to file - must be called after training/forward pass
    void save_model(const std::string filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("unable to open file for writing: " + filename);
        }

        // write number of layers
        uint32_t num_layers = static_cast<uint32_t>(layers.size());
        file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

        // write layer specifications
        for (const auto &layer : layers) {
            // write layer type
            uint32_t layer_type;
            if (dynamic_cast<ConvolutionLayer *>(layer.get())) {
                layer_type = 0;
            } else if (dynamic_cast<PoolingLayer *>(layer.get())) {
                layer_type = 1;
            } else if (dynamic_cast<DenseLayer *>(layer.get())) {
                layer_type = 2;
            }
            file.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));

            // write layer-specific parameters
            if (auto conv_layer = dynamic_cast<ConvolutionLayer *>(layer.get())) {
                uint32_t channels_in = static_cast<uint32_t>(conv_layer->channels_in);
                uint32_t out_channels = static_cast<uint32_t>(conv_layer->out_channels);
                uint32_t kernel_size = static_cast<uint32_t>(conv_layer->kernel_size);
                uint32_t mode_length = static_cast<uint32_t>(conv_layer->mode.length());

                file.write(reinterpret_cast<const char *>(&channels_in), sizeof(channels_in));
                file.write(reinterpret_cast<const char *>(&out_channels), sizeof(out_channels));
                file.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
                file.write(reinterpret_cast<const char *>(&mode_length), sizeof(mode_length));
                file.write(conv_layer->mode.c_str(), mode_length);

                // save weights and biases
                for (const auto &weight : conv_layer->weights) {
                    save_tensor(file, weight);
                }
                for (const auto &bias : conv_layer->biases) {
                    save_tensor(file, bias);
                }
            } else if (auto pool_layer = dynamic_cast<PoolingLayer *>(layer.get())) {
                uint32_t kernel_size = static_cast<uint32_t>(pool_layer->kernel_size);
                uint32_t stride = static_cast<uint32_t>(pool_layer->stride);
                uint32_t mode_length = static_cast<uint32_t>(pool_layer->mode.length());

                file.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
                file.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
                file.write(reinterpret_cast<const char *>(&mode_length), sizeof(mode_length));
                file.write(pool_layer->mode.c_str(), mode_length);
            } else if (auto dense_layer = dynamic_cast<DenseLayer *>(layer.get())) {
                uint32_t input_size = static_cast<uint32_t>(dense_layer->weights[0].width);
                uint32_t output_size = static_cast<uint32_t>(dense_layer->weights[0].height);
                uint32_t activation_length = static_cast<uint32_t>(dense_layer->activation_function.length());

                file.write(reinterpret_cast<const char *>(&input_size), sizeof(input_size));
                file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));
                file.write(reinterpret_cast<const char *>(&activation_length), sizeof(activation_length));
                file.write(dense_layer->activation_function.c_str(), activation_length);

                // save weights and biases
                for (const auto &weight : dense_layer->weights) {
                    save_tensor(file, weight);
                }
                for (const auto &bias : dense_layer->biases) {
                    save_tensor(file, bias);
                }
            }
        }
    }

    static NeuralNetwork load_model(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("unable to open file for reading: " + filename);
        }

        NeuralNetwork nn;

        // read number of layers
        uint32_t num_layers;
        file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

        // read and reconstruct each layer
        for (uint32_t i = 0; i < num_layers; ++i) {
            uint32_t layer_type;
            file.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));

            if (layer_type == 0) {  // ConvolutionLayer
                uint32_t channels_in, out_channels, kernel_size, mode_length;
                file.read(reinterpret_cast<char *>(&channels_in), sizeof(channels_in));
                file.read(reinterpret_cast<char *>(&out_channels), sizeof(out_channels));
                file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
                file.read(reinterpret_cast<char *>(&mode_length), sizeof(mode_length));

                std::string mode(mode_length, '\0');
                file.read(&mode[0], mode_length);

                // reconstruct convolution layer
                auto layer = std::make_unique<ConvolutionLayer>(channels_in, out_channels, kernel_size, mode);

                // load weights and biases
                for (auto &weight : layer->weights) {
                    load_tensor(file, weight);
                }
                for (auto &bias : layer->biases) {
                    load_tensor(file, bias);
                }

                nn.layers.push_back(std::move(layer));
            } else if (layer_type == 1) {  // PoolingLayer
                uint32_t kernel_size, stride, mode_length;
                file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
                file.read(reinterpret_cast<char *>(&stride), sizeof(stride));
                file.read(reinterpret_cast<char *>(&mode_length), sizeof(mode_length));

                std::string mode(mode_length, '\0');
                file.read(&mode[0], mode_length);

                nn.layers.push_back(std::make_unique<PoolingLayer>(kernel_size, stride, mode));
            } else if (layer_type == 2) {  // DenseLayer
                uint32_t input_size, output_size, activation_length;
                file.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));
                file.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));
                file.read(reinterpret_cast<char *>(&activation_length), sizeof(activation_length));

                std::string activation(activation_length, '\0');
                file.read(&activation[0], activation_length);

                // reconstruct dense layer
                auto layer = std::make_unique<DenseLayer>(input_size, output_size, activation);

                // load weights and biases
                for (auto &weight : layer->weights) {
                    load_tensor(file, weight);
                }
                for (auto &bias : layer->biases) {
                    load_tensor(file, bias);
                }

                nn.layers.push_back(std::move(layer));
            }
        }

        nn.layers_created = true;
        return nn;
    }

    // method for WASM interface for interactive demo
    std::vector<float> predict_digit(const std::vector<float> &input_data) {
        if (input_data.size() != 28 * 28) {
            throw std::runtime_error("input data must be a flat vector of 28x28 pixels");
        }
        // convert flat vector to Tensor3D (assuming MNIST 28x28 input)
        Tensor3D input(1, 28, 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input(0, i, j) = input_data[i * 28 + j];
            }
        }

        // get prediction
        Tensor3D output = feedforward(input);

        // convert output to vector
        std::vector<float> result(10);
        for (int i = 0; i < 10; i++) {
            result[i] = output(0, i, 0);
        }
        return result;
    }
};

// todo:
// implement other modes than same
// implement different strides in convlayer
// batch normalisation
// dropout

std::pair<std::vector<std::vector<Tensor3D>>, std::vector<std::vector<Tensor3D>>> load_mnist_data(std::string path_to_data) {
    // create training set from binary image data files

    std::vector<std::vector<Tensor3D>> training_set;
    training_set.reserve(9000);
    std::vector<std::vector<Tensor3D>> eval_set;
    eval_set.reserve(1000);

    for (int i = 0; i < 10; ++i) {
        std::string file_path = path_to_data + "/data" + std::to_string(i) + ".dat";
        std::vector<unsigned char> full_digit_data = read_file(file_path);
        assert(full_digit_data.size() == 784000);

        for (int j = 0; j < 784000; j += 28 * 28) {  // todo make more general with training ratio
            // create the input Tensor3D with shape (1, 28, 28)
            Tensor3D input_data(1, 28, 28);

            // fill the tensor with normalised pixel values
            for (int row = 0; row < 28; ++row) {
                for (int col = 0; col < 28; ++col) {
                    float normalised_pixel = static_cast<float>(full_digit_data[j + row * 28 + col]) / 255.0f;
                    input_data(0, row, col) = normalised_pixel;
                }
            }

            std::vector<float> data;

            // construct the label Tensor3D with 1.0 in the position of the digit and zeros elsewhere
            for (size_t l = 0; l < i; ++l) {
                data.push_back(0.0f);
            }
            data.push_back(1.0f);
            for (size_t l = 0; l + i + 1 < 10; ++l) {
                data.push_back(0.0f);
            }

            Tensor3D label_data(1, 10, 1, data);

            // push both image and label into appropriate set
            if (j < 705600) {
                training_set.push_back({input_data, label_data});
            } else {
                eval_set.push_back({input_data, label_data});
            }
        }
    }
    return {training_set, eval_set};
}

int main() {
    // load and split MNIST dataset
    auto [training_set, eval_set] = load_mnist_data("mnist-data");

    // create CNN architecture
    NeuralNetwork nn;
    nn.add_conv_layer(32, 3);     // 32 3x3 filters, same padding
    nn.add_conv_layer(32, 3);     // 32 3x3 filters, same padding
    nn.add_pool_layer();          // 2x2 max pooling
    nn.add_conv_layer(64, 3);     // 64 3x3 filters, same padding
    nn.add_conv_layer(64, 3);     // 64 3x3 filters, same padding
    nn.add_pool_layer();          // 2x2 max pooling
    nn.add_dense_layer(128);      // fully connected layer with 128 units
    nn.add_dense_layer(10, "softmax");  // output layer with softmax activation

    // set loss function and optimiser
    nn.set_loss(std::make_unique<CrossEntropyLoss>());
    nn.set_optimiser(std::make_unique<AdamWOptimiser>());

    // training parameters
    const int num_epochs = 100;
    const int batch_size = 200;

    // train network
    nn.train(training_set, eval_set, num_epochs, batch_size);
    std::cout << "Training complete\n" << std::endl;

    return 0;
}

// tensor3d.hpp
#ifndef TENSOR3D_HPP
#define TENSOR3D_HPP

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <stdexcept>

class Tensor3D {
private:
    std::vector<float> data;

public:
    size_t height, width, depth;

    // constructors
    Tensor3D();
    Tensor3D(size_t rows, size_t cols);
    Tensor3D(size_t depth, size_t height, size_t width);
    Tensor3D(size_t depth, size_t height, size_t width, const std::vector<float>& data);
    Tensor3D(size_t depth, size_t height, size_t width, 
             const std::vector<std::vector<std::vector<float>>>& data);

    // core methods
    size_t index(size_t d, size_t h, size_t w);
    const size_t index(size_t d, size_t h, size_t w) const;
    float& operator()(size_t d, size_t h, size_t w);
    const float& operator()(size_t d, size_t h, size_t w) const;
    Tensor3D operator()(size_t d);
    const Tensor3D operator()(size_t d) const;
    std::vector<float>& get_flat_data();
    const std::vector<float>& get_flat_data() const;

    // operations
    float dot_with_kernel_at_position(const Tensor3D& kernel, size_t start_x, size_t start_y) const;
    static Tensor3D pad(const Tensor3D& input, int amount = 1);

    // initialization
    void he_initialise();
    void xavier_initialise();
    void uniform_initialise(float lower_bound = 0.0f, float upper_bound = 1.0f);
    void zero_initialise();

    // operators
    Tensor3D operator*(const Tensor3D& other) const;
    Tensor3D operator+(const Tensor3D& other) const;
    Tensor3D operator+(const float& other) const;
    Tensor3D operator-(const Tensor3D& other) const;
    Tensor3D operator*(float scalar) const;
    Tensor3D hadamard(const Tensor3D& other) const;
    
    // transformations
    Tensor3D apply(float (*func)(float)) const;

    template <typename Func>
    Tensor3D apply(Func func) const {
        Tensor3D result(depth, height, width);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    Tensor3D transpose() const;
    Tensor3D softmax() const;
    Tensor3D flatten() const;
    Tensor3D unflatten(size_t new_depth, size_t new_height, size_t new_width) const;
    static Tensor3D Conv(const Tensor3D& input, const Tensor3D& kernel);
    Tensor3D rotate_180() const;
    void set_depth_slice(size_t depth_index, const Tensor3D& slice);

    // friend functions
    friend std::ostream& operator<<(std::ostream& os, const Tensor3D& tensor);

    // file operations
    void save_to_file(std::ofstream &file) const;
    void load_from_file(std::ifstream &file);
};

#endif

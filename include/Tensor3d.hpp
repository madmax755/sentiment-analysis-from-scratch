// Tensor3d.hpp
#ifndef Tensor3d_HPP
#define Tensor3d_HPP

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <stdexcept>

class Tensor3d {
private:
    std::vector<float> data;

public:
    size_t height, width, depth;

    // constructors
    Tensor3d();
    Tensor3d(size_t rows, size_t cols);
    Tensor3d(size_t depth, size_t height, size_t width);
    Tensor3d(size_t depth, size_t height, size_t width, const std::vector<float>& data);
    Tensor3d(size_t depth, size_t height, size_t width, 
             const std::vector<std::vector<std::vector<float>>>& data);

    // core methods
    size_t index(size_t d, size_t h, size_t w);
    const size_t index(size_t d, size_t h, size_t w) const;
    float& operator()(size_t d, size_t h, size_t w);
    const float& operator()(size_t d, size_t h, size_t w) const;
    Tensor3d operator()(size_t d);
    const Tensor3d operator()(size_t d) const;
    Tensor3d col(int index) const;
    std::vector<float>& get_flat_data();
    const std::vector<float>& get_flat_data() const;

    // operations
    float dot_with_kernel_at_position(const Tensor3d& kernel, size_t start_x, size_t start_y) const;
    static Tensor3d pad(const Tensor3d& input, int amount = 1);

    // initialization
    void he_initialise();
    void xavier_initialise();
    void uniform_initialise(float lower_bound = 0.0f, float upper_bound = 1.0f);
    void zero_initialise();

    // operators
    Tensor3d operator*(const Tensor3d& other) const;
    Tensor3d operator+(const Tensor3d& other) const;
    Tensor3d operator+(const float& other) const;
    Tensor3d operator-(const Tensor3d& other) const;
    Tensor3d operator*(float scalar) const;
    Tensor3d hadamard(const Tensor3d& other) const;
    
    // transformations
    Tensor3d apply(float (*func)(float)) const;

    template <typename Func>
    Tensor3d apply(Func func) const {
        Tensor3d result(depth, height, width);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    Tensor3d transpose() const;
    Tensor3d softmax(std::string dim = "height") const;
    Tensor3d flatten() const;
    Tensor3d unflatten(size_t new_depth, size_t new_height, size_t new_width) const;
    static Tensor3d Conv(const Tensor3d& input, const Tensor3d& kernel);
    Tensor3d rotate_180() const;
    Tensor3d diag() const;
    void set_depth_slice(size_t depth_index, const Tensor3d& slice);

    // friend functions
    friend std::ostream& operator<<(std::ostream& os, const Tensor3d& tensor);

    // file operations
    void save_to_file(std::ofstream &file) const;
    void load_from_file(std::ifstream &file);
    std::pair<float, float> get_magnitudes() const;
    void check_for_nans(const std::string& operation) const;
};

#endif

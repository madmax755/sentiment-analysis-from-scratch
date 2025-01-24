#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/Tensor3d.hpp"
#include "../include/tokeniser.hpp"

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

class GRUCell {
   public:
    size_t input_size;
    size_t hidden_size;

    // gate weights and biases
    Tensor3d W_z;  // update gate weights for input
    Tensor3d U_z;  // update gate weights for hidden state
    Tensor3d b_z;  // update gate bias

    Tensor3d W_r;  // reset gate weights for input
    Tensor3d U_r;  // reset gate weights for hidden state
    Tensor3d b_r;  // reset gate bias

    Tensor3d W_h;  // candidate hidden state weights for input
    Tensor3d U_h;  // candidate hidden state weights for hidden state
    Tensor3d b_h;  // candidate hidden state bias

    

    GRUCell(size_t input_size, size_t hidden_size)
        : input_size(input_size),
          hidden_size(hidden_size),
          W_z(hidden_size, input_size),
          U_z(hidden_size, hidden_size),
          b_z(hidden_size, 1),
          W_r(hidden_size, input_size),
          U_r(hidden_size, hidden_size),
          b_r(hidden_size, 1),
          W_h(hidden_size, input_size),
          U_h(hidden_size, hidden_size),
          b_h(hidden_size, 1) {
        // initialise weights using Xavier initialization
        W_z.xavier_initialise();
        U_z.xavier_initialise();
        W_r.xavier_initialise();
        U_r.xavier_initialise();
        W_h.xavier_initialise();
        U_h.xavier_initialise();

        // biases are initialised to zero by default
    }

    // forward pass that does not store states - used for evaluation so we dont clog up time_steps
    Tensor3d forward(const Tensor3d& x, const Tensor3d& h_prev) {
        // update gate
        Tensor3d z = (W_z * x + U_z * h_prev + b_z).apply(sigmoid);

        // reset gate
        Tensor3d r = (W_r * x + U_r * h_prev + b_r).apply(sigmoid);

        // candidate hidden state
        Tensor3d h_candidate = (W_h * x + U_h * (r.hadamard(h_prev)) + b_h).apply(std::tanh);

        // final hidden state
        Tensor3d h = z.hadamard(h_prev) + (z.apply([](float x) { return 1.0f - x; }).hadamard(h_candidate));

        return h;
    }
};

// layer class representing a single layer in the neural network
class Layer {
   public:
    Tensor3d weights;
    Tensor3d bias;
    std::string activation_function;

    /**
     * @brief Constructs a Layer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    Layer(size_t input_size, size_t output_size, std::string activation_function = "sigmoid")
        : weights(output_size, input_size), bias(output_size, 1), activation_function(activation_function) {
        if (activation_function == "sigmoid") {
            weights.xavier_initialise();
        } else if (activation_function == "relu") {
            weights.he_initialise();
        } else {
            weights.uniform_initialise();
        }
    }

    /**
     * @brief Performs feedforward operation for this layer.
     * @param input The input matrix.
     * @return The output matrix after applying the layer's transformation.
     */
    Tensor3d feedforward(const Tensor3d& input) {
        // compute pre-activation
        Tensor3d z = weights * input + bias;

        // apply activation function
        Tensor3d output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else if (activation_function == "none") {
            output = z;  // no activation
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return output;
    }
};

class AttentionLayer {
   public:
    Tensor3d weights;
    Tensor3d scoring_vector;
    size_t attention_size;
    size_t hidden_size;

    // assumes hidden_states are (1xhidden_sizex1)
    AttentionLayer(size_t hidden_size, size_t attention_size)
        : weights(attention_size, hidden_size),
          scoring_vector(1, attention_size),
          attention_size(attention_size),
          hidden_size(hidden_size) {
        weights.xavier_initialise();
        scoring_vector.xavier_initialise();
    }

    // assumes hidden_states are (1xhidden_sizex1)
    Tensor3d forward(const std::vector<Tensor3d>& hidden_states) {
        // check hidden_states are not empty
        if (hidden_states.empty()) {
            throw std::runtime_error("hidden_states must not be empty");
        }
        // check hidden_states are (1xhidden_sizex1)
        if (hidden_states[0].height != weights.width or hidden_states[0].width != 1) {
            throw std::runtime_error("hidden_states must be (1xhidden_sizex1)");
        }

        // concatenate hidden states into a (1xhidden_sizexno_hidden) tensor
        Tensor3d H(1, hidden_states[0].height, hidden_states.size());
        for (size_t i = 0; i < hidden_states.size(); ++i) {
            for (size_t j = 0; j < hidden_states[i].height; ++j) {
                H(0, j, i) = hidden_states[i](0, j, 0);
            }
        }

        // 1. Y = WH                   // pre-activation
        // 2. T = tanh(Y)              // tanh applied element-wise
        // 3. S = vT                   // raw scores
        // 4. A = softmax(S)           // attention weights
        // 5. C = HA^T (check this)      // context vector
        // store A, S, T for backprop

        // compute attention scores
        Tensor3d T = (weights * H).apply(std::tanh);
        Tensor3d S = scoring_vector * T;

        // across width dimension as S is (1, 1, no_hidden)
        Tensor3d A = S.softmax("width");

        Tensor3d C = H * A.transpose();
        return C;
    }
};

class MLP {
   public:
    std::vector<Layer> layers;

    /**
     * @brief Constructs a MLP object with the specified topology and activation functions.
     * @param topology A vector specifying the number of neurons in each layer.
     * @param activation_functions A vector specifying the activation function for each layer (optional).
     */

    MLP(const std::vector<int>& topology, const std::vector<std::string> activation_functions = {}) {
        if (topology.empty()) {
            throw std::invalid_argument("Topology cannot be empty");
        }
        for (int size : topology) {
            if (size <= 0) {
                throw std::invalid_argument("Layer size must be positive");
            }
        }
        if ((activation_functions.size() + 1 != topology.size()) and (activation_functions.size() != 0)) {
            throw std::invalid_argument(
                "the size of activations_functions vector must be the same size as no. layers (ex. input)");
        } else if (activation_functions.size() == 0) {
            for (size_t i = 1; i < topology.size(); i++) {
                // do not pass in specific activation function - use the default specified in the layer constructor
                layers.emplace_back(topology[i - 1], topology[i]);
            }
        } else {
            for (size_t i = 1; i < topology.size(); i++) {
                layers.emplace_back(topology[i - 1], topology[i], activation_functions[i - 1]);
            }
        }
    }

    /**
     * @brief Performs feedforward operation through all layers of the network.
     * @param input The input matrix.
     * @return The output matrix after passing through all layers.
     */
    Tensor3d feedforward(const Tensor3d& input) {
        Tensor3d current = input;
        for (auto& layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
    }

    size_t get_index_of_max_element_in_nx1_matrix(const Tensor3d& matrix) const {
        size_t index = 0;
        float max_value = matrix(1, 0, 0);
        for (size_t i = 1; i < matrix.height; ++i) {
            if (matrix(1, i, 0) > max_value) {
                index = i;
                max_value = matrix(1, i, 0);
            }
        }
        return index;
    }
};

class Predictor {
   private:
    GRUCell gru;
    AttentionLayer attention;
    MLP mlp;

    size_t input_size;
    size_t hidden_size;
    size_t attention_size;
    size_t output_size;

    Tokeniser tokeniser;
   public:
    Predictor(const std::string model_path, const std::string embeddings_path) 
        : gru(1, 1),                   // temporary values, will be overwritten
          attention(1, 1),             // temporary values, will be overwritten
          mlp({1}),                    // temporary values, will be overwritten
          input_size(0), hidden_size(0),
          attention_size(0), output_size(0),
          tokeniser(embeddings_path) {
        
        Predictor loaded = Predictor::load_model(model_path);
        
        // copy all members except tokeniser
        this->gru = loaded.gru;
        this->attention = loaded.attention;
        this->mlp = loaded.mlp;
        this->input_size = loaded.input_size;
        this->hidden_size = loaded.hidden_size;
        this->attention_size = loaded.attention_size;
        this->output_size = loaded.output_size;
    }

    Predictor(size_t input_size, size_t hidden_size, size_t attention_size, size_t output_size, std::vector<int> mlp_topology,
              std::vector<std::string> mlp_activation_functions = {})
        : gru(input_size, hidden_size),
          attention(hidden_size, attention_size),
          mlp(mlp_topology, mlp_activation_functions),
          input_size(input_size),
          hidden_size(hidden_size),
          attention_size(attention_size),
          output_size(output_size) {}

    // process sequence and return prediction (full feedforward pass)
    float predict(std::string input) {

        std::vector<Tensor3d> input_sequence = tokeniser.string_to_embeddings(input);

        // initialise hidden state
        Tensor3d h_t(hidden_size, 1);

        // process sequence through GRU
        std::vector<Tensor3d> h_states;
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
            h_states.push_back(h_t);
        }

        // process through attention layer
        h_t = attention.forward(h_states);

        // final linear layer
        return mlp.feedforward(h_t)(0,0,0);
    }

    static Predictor load_model(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("could not open file for loading: " + filepath);
        }

        // load model architecture parameters
        size_t input_size, hidden_size, output_size, attention_size;
        file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        file.read(reinterpret_cast<char*>(&attention_size), sizeof(attention_size));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

        // create predictor with loaded dimensions
        std::vector<int> mlp_topology = {static_cast<int>(hidden_size)};  // will be cleared and populated later just necessary to initialise MLP 
        Predictor predictor(input_size, hidden_size, attention_size, output_size, mlp_topology);  

        // load GRU parameters
        predictor.gru.W_z.load_from_file(file);
        predictor.gru.U_z.load_from_file(file);
        predictor.gru.b_z.load_from_file(file);
        predictor.gru.W_r.load_from_file(file);
        predictor.gru.U_r.load_from_file(file);
        predictor.gru.b_r.load_from_file(file);
        predictor.gru.W_h.load_from_file(file);
        predictor.gru.U_h.load_from_file(file);
        predictor.gru.b_h.load_from_file(file);

        // load attention parameters
        predictor.attention.weights.load_from_file(file);
        predictor.attention.scoring_vector.load_from_file(file);

        // load number of MLP layers
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // clear existing layers and load new ones
        predictor.mlp.layers.clear();

        // load MLP parameters
        for (size_t i = 0; i < num_layers; ++i) {
            Layer layer(1, 1);  // temporary dimensions, will be overwritten
            layer.weights.load_from_file(file);
            layer.bias.load_from_file(file);

            // load activation function name
            size_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
            std::vector<char> name_buffer(name_length);
            file.read(name_buffer.data(), name_length);
            layer.activation_function = std::string(name_buffer.data(), name_length);

            predictor.mlp.layers.push_back(layer);
        }

        return predictor;
    }
};

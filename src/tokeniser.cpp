#include "../include/tokeniser.hpp"
#include "../include/tensor3d.hpp"

Tokeniser::Tokeniser(const std::string& glove_path) {
    loadGloveEmbeddings(glove_path);
}

std::vector<Tensor3D> Tokeniser::string_to_embeddings(const std::string& text) {
    std::vector<std::string> tokens = tokenise(text);
    std::vector<Tensor3D> embeddings;
    for (const std::string& token : tokens) {
        embeddings.push_back(Tensor3D(1, embedding_dim, 1, getEmbedding(token)));
    }
    return embeddings;
}

// simple word tokenisation
std::vector<std::string> Tokeniser::tokenise(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    
    // convert to lowercase and split on spaces/punctuation
    for (char c : text) {
        if (std::isalnum(c)) {
            word += std::tolower(c);
        } else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) {
        tokens.push_back(word);
    }
    return tokens;
}

// get embedding for a single word
std::vector<float> Tokeniser::getEmbedding(const std::string& word) {
    auto it = embeddings.find(word);
    return it != embeddings.end() ? it->second : unk_embedding;
}

void Tokeniser::loadGloveEmbeddings(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("could not open embeddings file: " + path);
    }

    std::string line;
    
    // read glove file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        
        std::vector<float> vector(embedding_dim);
        for (int i = 0; i < embedding_dim; i++) {
            iss >> vector[i];
        }
        embeddings[word] = vector;
    }

    // initialise unknown token as zeros (could use random/mean)
    unk_embedding = std::vector<float>(embedding_dim, 0.0f);
}

// tokeniser.hpp

#ifndef TOKENISER_HPP
#define TOKENISER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include "tensor3d.hpp"
class Tokeniser {
    private:
        // map to store word -> embedding vector
        std::unordered_map<std::string, std::vector<float>> embeddings;
        const int embedding_dim = 100;
        std::vector<float> unk_embedding;

    public:
        Tokeniser(const std::string& glove_path);

        // tokenise text and convert to embeddings
        std::vector<Tensor3D> string_to_embeddings(const std::string& text);

        // simple word tokenisation
        std::vector<std::string> tokenise(const std::string& text);

        // get embedding for a single word
        std::vector<float> getEmbedding(const std::string& word);

    private:
        void loadGloveEmbeddings(const std::string& path);
};

#endif
#include "../include/tokeniser.hpp"
#include "../include/Tensor3d.hpp"

Tokeniser::Tokeniser(const std::string& glove_path) {
    loadGloveEmbeddings(glove_path);
}

std::vector<Tensor3d> Tokeniser::string_to_embeddings(const std::string& text) {
    std::vector<std::string> tokens = tokenise(text);
    std::vector<Tensor3d> embeddings;
    for (const std::string& token : tokens) {
        embeddings.push_back(Tensor3d(1, embedding_dim, 1, getEmbedding(token)));
    }
    return embeddings;
}

// clean input text
std::string Tokeniser::clean_text(const std::string& text) {
    std::string cleaned_text;
    bool space = false;
    for (char c : text) {
        if (std::isspace(c)) {
            // remove multiple spaces
            if (!space) {
                cleaned_text += ' ';
                space = true;
            }
        } else if (std::isalnum(c) or c == '.' or c == '!' or c == '?' or c == '\'') {
            // keep alphanumeric characters, punctuation, and apostrophes
            cleaned_text += c;
            space = false;
        }
    }
    // remove trailing space if present
    if (!cleaned_text.empty() && cleaned_text.back() == ' ') {
        cleaned_text.pop_back();
    }
    return cleaned_text;
}

// simple word tokenisation
std::vector<std::string> Tokeniser::tokenise(const std::string& text) {
    std::string cleaned_text = clean_text(text);  // clean the text before tokenising
    std::vector<std::string> tokens;
    std::string word;

    // split on spaces/punctuation
    for (char c : cleaned_text) {
        if (std::isalnum(c)) {
            word += c;
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

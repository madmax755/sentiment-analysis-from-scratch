#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

class Tokeniser {
    private:
        // map to store word -> embedding vector
        std::unordered_map<std::string, std::vector<float>> embeddings;
        const int embedding_dim = 100;
        std::vector<float> unk_embedding;

    public:
        Tokeniser(const std::string& glove_path);

        // simple word tokenisation
        std::vector<std::string> tokenise(const std::string& text);

        // get embedding for a single word
        std::vector<float> getEmbedding(const std::string& word);

    private:
        void loadGloveEmbeddings(const std::string& path);
};

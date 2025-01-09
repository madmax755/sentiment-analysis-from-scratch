#include "../include/tensor3d.hpp"
#include "../include/tokeniser.hpp"

int main() {

    std::cout << "loading embeddings..." << std::endl;
    Tokeniser tokeniser("../data/glove.6B.100d.txt");
    std::cout << "tokenising..." << std::endl;
    std::vector<std::string> tokens = tokeniser.tokenise("lol thats swag");
    for (std::string token : tokens) {
        std::cout << token << ":\n";
        std::vector<float> embedding = tokeniser.getEmbedding(token);
        for (float value : embedding) {
            std::cout << value << " ";
        }
        std::cout << "\n\n";
    }

    return 0;
}
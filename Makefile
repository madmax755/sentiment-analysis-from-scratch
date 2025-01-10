CXX = g++
CXXFLAGS = -O3 -I./include -std=c++20
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Find all cpp files and generate object file names
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

$(BIN_DIR)/main.exe: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(CXXFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean run rebuild

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/main.exe

run: $(BIN_DIR)/main.exe
	cd $(BIN_DIR) && ./main.exe

rebuild: clean $(BIN_DIR)/main.exe run
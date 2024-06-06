CPU_DIR =  .
BUILD_DIR = .
GCC = g++
CPP_FLAGS = -march=haswell -O2 -std=c++11 -g -mfma
BIN = ./CPU
CPU:
	@mkdir -p $(BUILD_DIR)
	@$(GCC)  $(CPU_DIR)/CPU.cpp $(CPP_FLAGS) -o $(BUILD_DIR)/CPU
run:
	@$(BIN)
clean:
	@rm $(BUILD_DIR)/CPU
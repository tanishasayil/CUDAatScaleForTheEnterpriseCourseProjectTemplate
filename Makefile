NVCC = /usr/local/cuda/bin/nvcc

CXXFLAGS = -std=c++11
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

SRC = src/image_processing.cu
TARGET = bin/image_processing.exe

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p bin
	$(NVCC) $(SRC) $(CXXFLAGS) $(OPENCV_FLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f bin/image_processing.exe bin/lena_grayscale.png

help:
	@echo "make       - Build the CUDA image processing project"
	@echo "make run   - Build and run the project"
	@echo "make clean - Remove generated files"

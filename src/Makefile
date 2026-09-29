NVCC = nvcc
TARGET = bin/image_processing
SRC = src/image_processing.cu

all:
	mkdir -p bin
	$(NVCC) -std=c++11 $(SRC) -o $(TARGET)

clean:
	rm -rf bin/*

run:
	./$(TARGET) 100 256 256

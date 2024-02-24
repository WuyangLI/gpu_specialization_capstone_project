# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: src/linear_algebra_mnist_classifier.cu
	$(CXX) src/linear_algebra_mnist_classifier.cu --std c++17 `pkg-config opencv --cflags --libs` -o linear_algebra_mnist_classifier.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas -DDEBUG_CUDA

run: ./linear_algebra_mnist_classifier.exe

clean:
	rm -f *.exe output*.txt 
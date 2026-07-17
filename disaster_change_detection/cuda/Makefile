NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70 -allow-unsupported-compiler # Adjust architecture flag based on RU HPC GPU specs

TARGET = cuda_change_detector
SRC = change_detector.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
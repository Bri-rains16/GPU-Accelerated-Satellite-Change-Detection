CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
LDFLAGS = `pkg-config --cflags --libs opencv4`

TARGET = omp_preprocessing
SRC = omp_preprocessing.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
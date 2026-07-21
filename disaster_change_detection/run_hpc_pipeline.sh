#!/bin/bash

echo "Initializing Ramanujan Universe HPC Pipeline..."

#Activate Python Environment
source venv/bin/activate

#Phase 4: Compile and Run OpenMP
echo "--- Compiling OpenMP (Multi-Core CPU) ---"
cd openmp
make clean
make
python run_openmp.py
cd ..

#Phase 5: Compile and Run CUDA
echo "--- Compiling CUDA (NVIDIA A100 GPUs) ---"
cd cuda
make clean
make
python run_cuda.py
cd ..

#Phase 8: Generate Final Analytics
echo "--- Generating Performance Analytics ---"
cd benchmarks
python generate_charts.py
cd ..

echo "HPC Pipeline Execution Complete. Check the 'results' folder for your speedup charts."
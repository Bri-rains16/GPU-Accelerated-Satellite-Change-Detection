#!/bin/bash

# ==========================================
# RAMANUJAN UNIVERSE - MASTER EXECUTION JOB
# ==========================================

echo "Initializing Ramanujan Universe HPC Pipeline..."

# Move to submission directory (for PBS batch jobs)
# if [ -n "$PBS_O_WORKDIR" ]; then
#     cd "$PBS_O_WORKDIR"
# fi

# 1. Load the required HPC Modules (These commands may vary slightly based on RU's specific module names)
# module load python/3.10
# module load gcc/11.2.0    # For OpenMP
# module load cuda/12.1     # For nvcc and A100 GPUs

# 2. Activate Python Environment
source venv/bin/activate

# 3. Phase 4: Compile and Run OpenMP
echo "--- Compiling OpenMP (Multi-Core CPU) ---"
cd openmp
make clean
make
python run_openmp.py
cd ..

# 4. Phase 5: Compile and Run CUDA
echo "--- Compiling CUDA (NVIDIA A100 GPUs) ---"
cd cuda
make clean
make
python run_cuda.py
cd ..

# 5. Phase 8: Generate Final Analytics
echo "--- Generating Performance Analytics ---"
cd benchmarks
python generate_charts.py
cd ..

echo "HPC Pipeline Execution Complete. Check the 'results' folder for your speedup charts."
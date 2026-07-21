import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

def plot_performance_metrics():
    logger = setup_logger("HPC_Project")
    logger.info("Generating Comparative Analytics Visualizations...")

    omp_csv = "../results/openmp_benchmark.csv"
    cuda_csv = "../results/cuda_benchmark.csv"
    output_img = "../results/hpc_speedup_analysis.png"

    if not os.path.exists(omp_csv) or not os.path.exists(cuda_csv):
        logger.error("Benchmark data tracking logs missing in results directory.")
        return

    df_omp = pd.read_csv(omp_csv)
    df_cuda = pd.read_csv(cuda_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(df_omp['Threads'], df_omp['Speedup'], marker='o', color='crimson', linewidth=2, label='Measured Speedup')
    ax1.plot(df_omp['Threads'], df_omp['Threads'], linestyle='--', color='gray', label='Ideal Scaling (Linear)')
    ax1.set_title('OpenMP CPU Multi-Core Scalability Profile', fontsize=12, fontweight='bold')
    ax1.set_xlabel('CPU Thread Count Allocated', fontsize=10)
    ax1.set_ylabel('Calculated Speedup Factor', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    implementations = ['Sequential CPU Baseline', 'OpenMP (8 Threads)', 'CUDA GPU Custom Kernel']
    execution_times = [
        df_cuda.loc[df_cuda['Implementation'] == 'Sequential Baseline', 'Execution_Time_sec'].values[0],
        df_omp.loc[df_omp['Threads'] == 8, 'Execution_Time_sec'].values[0],
        df_cuda.loc[df_cuda['Implementation'] == 'CUDA Custom Kernel', 'Execution_Time_sec'].values[0]
    ]

    colors = ['navy', 'teal', 'darkorange']
    bars = ax2.bar(implementations, execution_times, color=colors, width=0.5)
    ax2.set_title('Cross-Architectural Execution Time Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Execution Latency (Seconds)', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    logger.info(f"Analytical charts rendered and exported successfully to {output_img}")

if __name__ == "__main__":
    plot_performance_metrics()
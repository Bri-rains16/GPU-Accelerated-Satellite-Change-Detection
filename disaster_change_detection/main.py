import os
import yaml
import argparse
import torch
from utils.logger import setup_logger

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():

    logger = setup_logger()
    logger.info("Initializing GPU-Accelerated Satellite Image Change Detection Pipeline...")

    config = load_config()
    logger.info(f"Loaded configuration for project: {config['project']['name']}")

    #Hardware Availability
    device = torch.device("cuda" if torch.cuda.is_available() and config['execution']['use_gpu_if_available'] else "cpu")
    logger.info(f"Primary Deep Learning Execution Device determined as: {device}")
    
    if device.type == 'cpu':
        logger.warning("CUDA is not available locally. Development will proceed on CPU. CUDA code will still be written for future HPC compilation.")

    #Pipeline Execution Controller
    if config['execution']['run_preprocessing']:
        logger.info("Phase 2 Placeholder: Executing Preprocessing Pipeline...")
        #preprocessing.run(config)
        
    if config['execution']['run_sequential']:
        logger.info("Phase 3 Placeholder: Executing Sequential Baseline...")
        
    if config['execution']['run_openmp']:
        logger.info("Phase 4 Placeholder: Executing OpenMP Acceleration...")
        
    if config['execution']['run_cuda']:
        logger.info("Phase 5 Placeholder: Executing CUDA Custom Kernels...")
        
    if config['execution']['run_deep_learning']:
        logger.info("Phase 6 Placeholder: Executing Deep Learning Pipeline...")
    logger.info("Pipeline execution finished gracefully.")
if __name__ == "__main__":
    main()
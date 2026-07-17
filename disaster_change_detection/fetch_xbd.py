import os
import sys
import argparse
import logging
import zipfile
import tarfile
import re

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("xBD_Fetcher")

def print_auth_instructions():
    instructions = """
================================================================================
                        xBD DATASET AUTHENTICATION INFO
================================================================================
To download the xBD dataset using this script, you have two primary methods:

METHOD 1: Kaggle API (Recommended for Automated Downloads)
1. Ensure your 'kaggle.json' token is placed in the correct directory:
   - Linux/macOS/RU HPC: ~/.kaggle/kaggle.json
   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
2. Secure the file permissions (Linux/macOS/RU HPC):
   chmod 600 ~/.kaggle/kaggle.json
3. Run this script:
   python fetch_xbd.py --method kaggle --dataset robikscube/xbd-dataset

METHOD 2: Manual Download (For xView2 Website / Pre-downloaded Archives)
1. Download the tarballs/zips.
2. Transfer them to the HPC.
3. Run this script to extract and filter only volcano & hurricane images:
   python fetch_xbd.py --method local --archive /path/to/downloaded_archive.tar.gz

STORAGE WARNING:
- The total raw download is 12-15 GB (filtered for volcano and hurricane).
- Due to the overlapping stride (128) in dataset_builder.py, the generated patches
  will expand to 30-40 GB.
- Ensure the HPC account has at least 60 GB of storage quota available!
================================================================================
"""
    print(instructions)

def create_dirs(output_dir):
    pre_dir = os.path.join(output_dir, "pre_disaster")
    post_dir = os.path.join(output_dir, "post_disaster")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    return pre_dir, post_dir

def download_from_kaggle(dataset_name, download_dir):
    logger.info("Initializing Kaggle API download...")
    try:
        import kaggle
    except ImportError:
        logger.error("The 'kaggle' python package is not installed. Please run: pip install kaggle")
        sys.exit(1)
        
    os.makedirs(download_dir, exist_ok=True)
    
    # Download files using Kaggle API. It automatically handles resuming/retries.
    logger.info(f"Downloading Kaggle dataset '{dataset_name}' to {download_dir}...")
    try:
        # Download as zip
        kaggle.api.dataset_download_files(dataset_name, path=download_dir, unzip=False, quiet=False)
        dataset_slug = dataset_name.split('/')[-1]
        archive_path = os.path.join(download_dir, f"{dataset_slug}.zip")
        if os.path.exists(archive_path):
            logger.info(f"Successfully downloaded archive to {archive_path}")
            return archive_path
        else:
            zips = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".zip")]
            if zips:
                return zips[0]
            raise FileNotFoundError("Could not find downloaded zip archive.")
    except Exception as e:
        logger.error(f"Failed to download dataset from Kaggle: {e}")
        sys.exit(1)

def filter_and_extract(archive_path, output_dir):
    logger.info(f"Filtering and extracting from: {archive_path}")
    pre_dir, post_dir = create_dirs(output_dir)
    
    pattern = re.compile(r'(volcano|hurricane)', re.IGNORECASE)
    
    extracted_count = 0
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                filename = os.path.basename(member.filename)
                
                # Exclusively png images for volcano/hurricane pre/post disaster
                if filename.endswith('.png') and pattern.search(filename):
                    if 'pre_disaster' in filename:
                        target_path = os.path.join(pre_dir, filename)
                        with open(target_path, 'wb') as f:
                            f.write(zip_ref.read(member))
                        extracted_count += 1
                    elif 'post_disaster' in filename:
                        target_path = os.path.join(post_dir, filename)
                        with open(target_path, 'wb') as f:
                            f.write(zip_ref.read(member))
                        extracted_count += 1
                        
    elif archive_path.endswith(('.tar.gz', '.tgz', '.tar')):
        mode = 'r:gz' if archive_path.endswith(('.tar.gz', '.tgz')) else 'r'
        with tarfile.open(archive_path, mode) as tar_ref:
            for member in tar_ref.getmembers():
                if not member.isfile():
                    continue
                filename = os.path.basename(member.name)
                
                if filename.endswith('.png') and pattern.search(filename):
                    if 'pre_disaster' in filename:
                        target_path = os.path.join(pre_dir, filename)
                        f_in = tar_ref.extractfile(member)
                        if f_in:
                            with open(target_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                            extracted_count += 1
                    elif 'post_disaster' in filename:
                        target_path = os.path.join(post_dir, filename)
                        f_in = tar_ref.extractfile(member)
                        if f_in:
                            with open(target_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                            extracted_count += 1
    else:
        logger.error(f"Unsupported archive format: {archive_path}. Only .zip and .tar.gz are supported.")
        sys.exit(1)
        
    logger.info(f"Extracted and filtered {extracted_count} images matching volcano/hurricane events.")

def verify_pairs(output_dir):
    logger.info("Verifying image pairs (pre_disaster and post_disaster)...")
    pre_dir = os.path.join(output_dir, "pre_disaster")
    post_dir = os.path.join(output_dir, "post_disaster")
    
    pre_files = {os.path.basename(f).replace('_pre_disaster.png', ''): f for f in os.listdir(pre_dir) if f.endswith('.png')}
    post_files = {os.path.basename(f).replace('_post_disaster.png', ''): f for f in os.listdir(post_dir) if f.endswith('.png')}
    
    mismatches = 0
    matched_pairs = 0
    
    for key, pre_path in pre_files.items():
        if key not in post_files:
            logger.warning(f"Unmatched pre-disaster image: {pre_path}")
            mismatches += 1
        else:
            matched_pairs += 1
            
    for key, post_path in post_files.items():
        if key not in pre_files:
            logger.warning(f"Unmatched post-disaster image: {post_path}")
            mismatches += 1
            
    logger.info(f"Verification complete: {matched_pairs} matching pairs found. {mismatches} mismatches found.")
    if mismatches > 0:
        logger.warning("Please make sure the dataset is fully downloaded and extracted. Mismatches will cause errors.")
    else:
        logger.info("All image pairs match perfectly!")

def main():
    parser = argparse.ArgumentParser(description="Fetch and filter xBD dataset for volcano and hurricane events.")
    parser.add_argument("--method", choices=["kaggle", "local"], default="kaggle", help="Download method")
    parser.add_argument("--dataset", default="robikscube/xbd-dataset", help="Kaggle dataset slug")
    parser.add_argument("--archive", help="Path to local archive (if method=local)")
    parser.add_argument("--output-dir", default="data/raw_xbd", help="Directory to save extracted files")
    parser.add_argument("--download-dir", default="data/downloads", help="Directory to save raw downloads")
    parser.add_argument("--info", action="store_true", help="Print auth instructions")
    
    args = parser.parse_args()
    
    if args.info:
        print_auth_instructions()
        sys.exit(0)
        
    logger.info("Starting Scoped xBD Dataset Fetcher...")
    
    if args.method == "kaggle":
        archive_path = download_from_kaggle(args.dataset, args.download_dir)
        filter_and_extract(archive_path, args.output_dir)
    elif args.method == "local":
        if not args.archive:
            logger.error("--archive is required when using --method local")
            sys.exit(1)
        filter_and_extract(args.archive, args.output_dir)
        
    verify_pairs(args.output_dir)
    logger.info("Successfully finished processing.")

if __name__ == "__main__":
    main()

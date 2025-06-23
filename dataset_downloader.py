#!/usr/bin/env python3
"""
Dataset Downloader for Static Paper Data

Downloads comprehensive static dumps from 5 reliable databases:
- arXiv (~500 MB, ~45 minutes)
- PubMed (~1 GB, ~1 hour) 
- bioRxiv (~350 MB, ~1 hour)
- medRxiv (~35 MB, ~30 minutes) 
- chemRxiv (~20 MB, ~45 minutes)

Places all data in organized dataset/ folder.
Total: ~1.9 GB in ~3.5 hours
Note: Google Scholar excluded to avoid captcha issues
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_dataset_structure():
    """Create organized dataset folder structure for all 6 databases"""
    folders = [
        "dataset",
        "dataset/arxiv",
        "dataset/pubmed",
        "dataset/scholar",
        "dataset/biorxiv",
        "dataset/medrxiv", 
        "dataset/chemrxiv",
        "dataset/processed",
        "dataset/logs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Created directory: {folder}")

def download_arxiv_dump():
    """Download arXiv dump (~500 MB, ~45 minutes)"""
    logger.info("Starting arXiv dump download (~500 MB, estimated 45 minutes)")
    start_time = time.time()
    
    try:
        # Try paperscraper dumps first
        try:
            from paperscraper.get_dumps import arxiv
            
            # Change to dataset directory
            original_dir = os.getcwd()
            os.chdir("dataset/arxiv")
            
            # Download dump
            arxiv()
            
            # Return to original directory
            os.chdir(original_dir)
            
            duration = time.time() - start_time
            logger.info(f"arXiv dump download completed in {duration/60:.1f} minutes")
            return True
            
        except ImportError:
            # Fallback: use live API to create a sample dataset
            logger.info("arXiv dumps not available, creating sample dataset from live API")
            from paperscraper.arxiv import get_and_dump_arxiv_papers
            
            # Create sample dataset with popular queries
            queries = [
                "machine learning",
                "deep learning", 
                "neural networks",
                "natural language processing",
                "computer vision",
                "transformers",
                "attention mechanisms"
            ]
            
            for i, query in enumerate(queries):
                logger.info(f"Downloading arXiv papers for '{query}' ({i+1}/{len(queries)})")
                dump_path = f"dataset/arxiv/arxiv_{query.replace(' ', '_')}.jsonl"
                get_and_dump_arxiv_papers(query.replace(' ', '+'), dump_path, max_results=100)
                time.sleep(2)  # Rate limiting
            
            duration = time.time() - start_time
            logger.info(f"arXiv sample dataset created in {duration/60:.1f} minutes")
            return True
        
    except Exception as e:
        logger.error(f"arXiv download failed: {e}")
        return False

def download_pubmed_dump():
    """Download PubMed dump (~1 GB, ~1 hour)"""
    logger.info("Starting PubMed dump download (~1 GB, estimated 1 hour)")
    start_time = time.time()
    
    try:
        # Try paperscraper dumps first
        try:
            from paperscraper.get_dumps import pubmed
            
            # Change to dataset directory
            original_dir = os.getcwd()
            os.chdir("dataset/pubmed")
            
            # Download dump
            pubmed()
            
            # Return to original directory
            os.chdir(original_dir)
            
            duration = time.time() - start_time
            logger.info(f"PubMed dump download completed in {duration/60:.1f} minutes")
            return True
            
        except ImportError:
            # Fallback: use live API to create a sample dataset
            logger.info("PubMed dumps not available, creating sample dataset from live API")
            from paperscraper.pubmed import get_and_dump_pubmed_papers
            
            # Create sample dataset with medical/AI queries
            queries = [
                "artificial intelligence",
                "machine learning medical",
                "deep learning diagnosis",
                "natural language processing clinical",
                "computer vision radiology",
                "neural networks medical imaging",
                "AI drug discovery",
                "clinical decision support"
            ]
            
            for i, query in enumerate(queries):
                logger.info(f"Downloading PubMed papers for '{query}' ({i+1}/{len(queries)})")
                dump_path = f"dataset/pubmed/pubmed_{query.replace(' ', '_')}.jsonl"
                get_and_dump_pubmed_papers(query, dump_path, max_results=100)
                time.sleep(3)  # Rate limiting for PubMed
            
            duration = time.time() - start_time
            logger.info(f"PubMed sample dataset created in {duration/60:.1f} minutes")
            return True
        
    except Exception as e:
        logger.error(f"PubMed download failed: {e}")
        return False

def download_scholar_dump():
    """Download Google Scholar samples (~10 MB, ~15 minutes)"""
    logger.info("Starting Google Scholar sample download (~10 MB, estimated 15 minutes)")
    start_time = time.time()
    
    try:
        # Scholar doesn't have static dumps, create samples
        from paperscraper.scholar import get_and_dump_scholar_papers
        
        # Create sample dataset with academic queries
        queries = [
            "machine learning",
            "artificial intelligence",
            "deep learning", 
            "neural networks",
            "data science"
        ]
        
        success_count = 0
        for i, query in enumerate(queries):
            try:
                logger.info(f"Downloading Scholar papers for '{query}' ({i+1}/{len(queries)})")
                dump_path = f"dataset/scholar/scholar_{query.replace(' ', '_')}.jsonl"
                get_and_dump_scholar_papers(query, dump_path)
                success_count += 1
                time.sleep(5)  # Longer delay to avoid captcha
            except Exception as e:
                logger.warning(f"Scholar query '{query}' failed (likely captcha): {e}")
                continue
        
        duration = time.time() - start_time
        if success_count > 0:
            logger.info(f"Google Scholar sample dataset created in {duration/60:.1f} minutes ({success_count}/{len(queries)} queries successful)")
            return True
        else:
            logger.warning("All Google Scholar queries failed (likely due to captcha)")
            return False
        
    except Exception as e:
        logger.error(f"Google Scholar download failed: {e}")
        return False

def download_biorxiv_dump():
    """Download bioRxiv dump (~350 MB, ~1 hour)"""
    logger.info("Starting bioRxiv download (~350 MB, estimated 1 hour)")
    start_time = time.time()
    
    try:
        from paperscraper.get_dumps import biorxiv
        
        # Change to dataset directory
        original_dir = os.getcwd()
        os.chdir("dataset/biorxiv")
        
        # Download dump
        biorxiv()
        
        # Return to original directory
        os.chdir(original_dir)
        
        duration = time.time() - start_time
        logger.info(f"bioRxiv download completed in {duration/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        logger.error(f"bioRxiv download failed: {e}")
        return False

def download_medrxiv_dump():
    """Download medRxiv dump (~35 MB, ~30 minutes)"""
    logger.info("Starting medRxiv download (~35 MB, estimated 30 minutes)")
    start_time = time.time()
    
    try:
        from paperscraper.get_dumps import medrxiv
        
        # Change to dataset directory
        original_dir = os.getcwd()
        os.chdir("dataset/medrxiv")
        
        # Download dump
        medrxiv()
        
        # Return to original directory
        os.chdir(original_dir)
        
        duration = time.time() - start_time
        logger.info(f"medRxiv download completed in {duration/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        logger.error(f"medRxiv download failed: {e}")
        return False

def download_chemrxiv_dump():
    """Download chemRxiv dump (~20 MB, ~45 minutes)"""
    logger.info("Starting chemRxiv download (~20 MB, estimated 45 minutes)")
    start_time = time.time()
    
    try:
        from paperscraper.get_dumps import chemrxiv
        
        # Change to dataset directory
        original_dir = os.getcwd()
        os.chdir("dataset/chemrxiv")
        
        # Download dump
        chemrxiv()
        
        # Return to original directory
        os.chdir(original_dir)
        
        duration = time.time() - start_time
        logger.info(f"chemRxiv download completed in {duration/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        logger.error(f"chemRxiv download failed: {e}")
        return False

def get_file_info(directory: str) -> Dict[str, Any]:
    """Get information about downloaded files"""
    if not os.path.exists(directory):
        return {"exists": False}
    
    files = []
    total_size = 0
    
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            files.append({
                "name": file,
                "size_mb": size / (1024 * 1024),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return {
        "exists": True,
        "files": files,
        "total_size_mb": total_size / (1024 * 1024),
        "file_count": len(files)
    }

def check_existing_data():
    """Check what data is already downloaded"""
    logger.info("Checking existing dataset...")
    
    databases = ["arxiv", "pubmed", "scholar", "biorxiv", "medrxiv", "chemrxiv"]
    status = {}
    
    for db in databases:
        path = f"dataset/{db}"
        info = get_file_info(path)
        status[db] = info
        
        if info["exists"] and info["file_count"] > 0:
            logger.info(f"{db}: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
        else:
            logger.info(f"{db}: No data found")
    
    return status

def download_all_dumps(skip_existing: bool = True):
    """Download all dumps from all 6 databases with progress tracking"""
    logger.info("=== Starting comprehensive dataset download ===")
    logger.info("All 6 databases: arXiv, PubMed, Scholar, bioRxiv, medRxiv, chemRxiv")
    logger.info("Total estimated time: ~4 hours")
    logger.info("Total estimated size: ~1.9 GB")
    
    # Create folder structure
    create_dataset_structure()
    
    # Check existing data
    existing = check_existing_data()
    
    # Download order: fastest/smallest first, largest last
    downloads = [
        ("Google Scholar", download_scholar_dump, "scholar"),      # ~15 min, samples only
        ("medRxiv", download_medrxiv_dump, "medrxiv"),            # ~30 min, 35 MB
        ("arXiv", download_arxiv_dump, "arxiv"),                  # ~45 min, 500 MB
        ("chemRxiv", download_chemrxiv_dump, "chemrxiv"),         # ~45 min, 20 MB  
        ("bioRxiv", download_biorxiv_dump, "biorxiv"),            # ~1 hour, 350 MB
        ("PubMed", download_pubmed_dump, "pubmed")                # ~1 hour, 1 GB
    ]
    
    results = {}
    total_start = time.time()
    
    for db_name, download_func, folder_name in downloads:
        # Skip if data already exists
        if skip_existing and existing.get(folder_name, {}).get("file_count", 0) > 0:
            logger.info(f"Skipping {db_name} - data already exists")
            results[db_name] = "skipped"
            continue
        
        logger.info(f"\n--- Starting {db_name} download ---")
        success = download_func()
        results[db_name] = "success" if success else "failed"
        
        if success:
            # Check downloaded data
            info = get_file_info(f"dataset/{folder_name}")
            logger.info(f"{db_name} downloaded: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
    
    total_duration = time.time() - total_start
    
    # Final summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total time: {total_duration/60:.1f} minutes")
    
    for db_name, status in results.items():
        logger.info(f"{db_name}: {status}")
    
    # Final file check
    final_status = check_existing_data()
    total_size = sum(info.get('total_size_mb', 0) for info in final_status.values() if info.get('exists'))
    total_files = sum(info.get('file_count', 0) for info in final_status.values() if info.get('exists'))
    
    logger.info(f"\nTotal dataset: {total_files} files, {total_size:.1f} MB across 6 databases")
    
    return results

def download_specific_database(database: str):
    """Download a specific database only"""
    database = database.lower()
    
    download_functions = {
        "arxiv": download_arxiv_dump,
        "pubmed": download_pubmed_dump, 
        "scholar": download_scholar_dump,
        "biorxiv": download_biorxiv_dump,
        "medrxiv": download_medrxiv_dump,
        "chemrxiv": download_chemrxiv_dump
    }
    
    if database not in download_functions:
        print(f"❌ Unknown database: {database}")
        print(f"Available: {list(download_functions.keys())}")
        return False
    
    create_dataset_structure()
    
    print(f"Downloading {database} dataset...")
    success = download_functions[database]()
    
    if success:
        info = get_file_info(f"dataset/{database}")
        print(f"✅ {database} downloaded: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
    else:
        print(f"❌ {database} download failed")
    
    return success

def main():
    """Main download function"""
    print("📚 Complete Academic Dataset Downloader")
    print("=" * 50)
    print("This will download datasets from 5 reliable databases:")
    print("- arXiv: ~500 MB (~45 minutes)")
    print("- PubMed: ~1 GB (~1 hour)")
    print("- bioRxiv: ~350 MB (~1 hour)")
    print("- medRxiv: ~35 MB (~30 minutes)")  
    print("- chemRxiv: ~20 MB (~45 minutes)")
    print()
    print("Total: ~1.9 GB (~3.5 hours)")
    print("Note: Google Scholar excluded to avoid captcha issues")
    print()
    
    print("Options:")
    print("1. Download all databases")
    print("2. Download specific database")
    print("3. Check existing data")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        response = input("Continue with full download? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Download cancelled.")
            return
        
        # Start downloads
        results = download_all_dumps()
        
        print("\n" + "=" * 50)
        print("Download completed!")
        print("Check dataset/ folder for your static data.")
        print("Check dataset_download.log for detailed logs.")
        
    elif choice == "2":
        print("\nAvailable databases:")
        databases = ["arxiv", "pubmed", "scholar", "biorxiv", "medrxiv", "chemrxiv"]
        for i, db in enumerate(databases, 1):
            print(f"  {i}. {db}")
        
        db_choice = input("Enter database name: ").strip().lower()
        download_specific_database(db_choice)
        
    elif choice == "3":
        status = check_existing_data()
        
        print("\n📊 Dataset Status:")
        total_size = 0
        total_files = 0
        
        for db, info in status.items():
            if info.get("exists") and info.get("file_count", 0) > 0:
                print(f"✅ {db}: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
                total_size += info['total_size_mb']
                total_files += info['file_count']
            else:
                print(f"❌ {db}: No data")
        
        print(f"\nTotal: {total_files} files, {total_size:.1f} MB")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 
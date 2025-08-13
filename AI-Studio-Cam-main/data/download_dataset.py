"""
Simple Dataset Image Downloader

Downloads images for specified classes and performs basic deduplication.

Usage:
------
python download_dataset.py
"""

import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image
import imagehash

# Configuration
IMAGES_PER_CLASS = 100
DATASET_DIR = 'dataset'

# Classes with their search terms
CLASSES = {
    'bottle': ['plastic water bottle', 'glass bottle', 'sports water bottle', 'water bottle on table', 'blue bottle'],
    'chair': ['office chair', 'wooden chair', 'plastic chair', 'desk chair', 'folding chair'],
    'laptop': ['open laptop', 'laptop on desk', 'closed laptop', 'silver laptop', 'black laptop computer'],
    'phone': ['smartphone on table', 'hand holding phone', 'black smartphone', 'mobile phone close up', 'white smartphone'],
    'book': ['open book on desk', 'stack of books', 'closed book', 'notebook on table', 'hardcover book'],
    'remote': ['tv remote', 'remote control on table', 'black remote', 'remote for tv', 'remote on couch'],
    'mouse': ['computer mouse', 'wireless mouse', 'mouse on desk', 'gaming mouse', 'black mouse'],
    'keyboard': ['computer keyboard', 'mechanical keyboard', 'wireless keyboard', 'keyboard on table', 'black keyboard'],
    'cup': ['coffee mug', 'tea cup on table', 'ceramic cup', 'glass cup', 'white mug'],
    'pen': ['blue pen', 'black pen', 'pen on notebook', 'ballpoint pen', 'writing pen']
}

def is_image_file(filename):
    """Check if file is an image"""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))

def deduplicate_images(class_dir):
    """Remove duplicate images using perceptual hashing"""
    files = [f for f in os.listdir(class_dir) if is_image_file(f)]
    hashes = {}
    duplicates = []
    
    for filename in files:
        filepath = os.path.join(class_dir, filename)
        try:
            with Image.open(filepath) as img:
                img_hash = str(imagehash.phash(img))
                if img_hash in hashes:
                    duplicates.append(filepath)
                else:
                    hashes[img_hash] = filepath
        except Exception:
            duplicates.append(filepath)
    
    # Remove duplicates
    for dup in duplicates:
        try:
            os.remove(dup)
        except Exception:
            pass

def download_images(class_name, target_count=100):
    """Download images for a specific class"""
    class_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Clean existing directory
    for f in os.listdir(class_dir):
        file_path = os.path.join(class_dir, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass
    
    print(f"Downloading images for class: {class_name}")
    
    # Get search terms for this class
    search_terms = CLASSES.get(class_name, [class_name])
    
    # Download images from all search terms
    for term in search_terms:
        print(f"  Downloading from: {term}")
        
        # Download from Google Images
        try:
            google_crawler = GoogleImageCrawler(storage={'root_dir': class_dir})
            google_crawler.crawl(keyword=term, max_num=200)
        except Exception:
            pass
        
        # Download from Bing Images
        try:
            bing_crawler = BingImageCrawler(storage={'root_dir': class_dir})
            bing_crawler.crawl(keyword=term, max_num=200)
        except Exception:
            pass
    
    # Remove duplicates
    deduplicate_images(class_dir)
    
    current_count = len([f for f in os.listdir(class_dir) if is_image_file(f)])
    print(f"  Downloaded {current_count} images total")
    
    # Limit to target count and rename
    files = sorted([f for f in os.listdir(class_dir) if is_image_file(f)])
    
    # Remove excess files
    for filename in files[target_count:]:
        try:
            os.remove(os.path.join(class_dir, filename))
        except Exception:
            pass
    
    # Rename files sequentially
    remaining_files = sorted([f for f in os.listdir(class_dir) if is_image_file(f)])
    for i, filename in enumerate(remaining_files, 1):
        old_path = os.path.join(class_dir, filename)
        ext = os.path.splitext(filename)[1]
        new_path = os.path.join(class_dir, f"image_{i:03d}{ext}")
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
            except Exception:
                pass
    
    final_count = len([f for f in os.listdir(class_dir) if is_image_file(f)])
    print(f"Downloaded {final_count} images for {class_name}")
    return final_count

if __name__ == '__main__':
    """Main function to download dataset"""
    print("Starting dataset download...")
    
    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    total_images = 0
    
    # Download images for each class
    for class_name in CLASSES:
        count = download_images(class_name, IMAGES_PER_CLASS)
        total_images += count
    
    print(f"\nDataset download complete!")
    print(f"Total images downloaded: {total_images}")
    
    # Print summary
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if is_image_file(f)])
            print(f"{class_name}: {count} images")
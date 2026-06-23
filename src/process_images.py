"""Post-process downloaded images: deduplicate, filter, report."""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

def process_images():
    CLEAN_DIR.mkdir(exist_ok=True)
    
    stats = defaultdict(lambda: {"total": 0, "duplicates": 0, "too_small": 0, "invalid": 0, "kept": 0})
    total_dup = 0
    total_invalid = 0
    total_small = 0
    total_kept = 0
    
    for cat_dir in sorted(RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        
        cat_name = cat_dir.name
        print(f"\n[{cat_name}]")
        
        # Scan images
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.gif'):
            for img_path in cat_dir.glob(ext):
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    if img.size[0] < 100 or img.size[1] < 100:
                        stats[cat_name]["too_small"] += 1
                        total_small += 1
                        continue
                    h = imagehash.phash(img)
                    images.append((img_path, h, img.size))
                except Exception:
                    stats[cat_name]["invalid"] += 1
                    total_invalid += 1
        
        stats[cat_name]["total"] = len(images)
        
        # Deduplicate: keep first of each hash group
        seen_hashes = {}
        unique = []
        for img_path, h, size in images:
            is_dup = False
            for seen_h in seen_hashes:
                if h - seen_h <= 8:  # Hamming distance threshold
                    stats[cat_name]["duplicates"] += 1
                    total_dup += 1
                    is_dup = True
                    break
            if not is_dup:
                seen_hashes[h] = img_path
                unique.append((img_path, size))
        
        # Copy unique images to clean dir
        cat_clean = CLEAN_DIR / cat_name
        cat_clean.mkdir(exist_ok=True)
        for img_path, size in unique:
            ext = img_path.suffix.lower()
            if ext == '.webp':
                # Convert webp to jpg
                out_name = img_path.stem + '.jpg'
            else:
                out_name = img_path.name
            shutil.copy2(img_path, cat_clean / out_name)
        
        stats[cat_name]["kept"] = len(unique)
        total_kept += len(unique)
        
        print(f"  Total: {len(images)}, Dup: {stats[cat_name]['duplicates']}, "
              f"Small: {stats[cat_name]['too_small']}, Invalid: {stats[cat_name]['invalid']}, "
              f"Kept: {len(unique)}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary: {total_kept} unique images from original ~{total_kept+total_dup}")
    print(f"  Duplicates removed: {total_dup}")
    print(f"  Too small: {total_small}")
    print(f"  Invalid/corrupt: {total_invalid}")
    
    # Per-category summary
    print(f"\n{'Category':<25s} {'Total':>6s} {'Dup':>5s} {'Small':>5s} {'Invalid':>7s} {'Kept':>5s}")
    print('-' * 58)
    total_all = sum(s['total'] for s in stats.values())
    for cat in sorted(stats.keys()):
        s = stats[cat]
        print(f"{cat:<25s} {s['total']:>6d} {s['duplicates']:>5d} "
              f"{s['too_small']:>5d} {s['invalid']:>7d} {s['kept']:>5d}")
    print('-' * 58)
    print(f"{'TOTAL':<25s} {total_all:>6d} {total_dup:>5d} {total_small:>5d} {total_invalid:>7d} {total_kept:>5d}")

if __name__ == "__main__":
    process_images()

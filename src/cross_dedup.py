"""Cross-category deduplication using perceptual hash."""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

RAW_DIR = Path(__file__).parent.parent / "data" / "clean"

def cross_dedup(hamming_threshold=6):
    """Deduplicate across all categories. Keep first occurrence only."""
    
    # Phase 1: Scan all images and compute hashes
    print("[1] Scanning all images...")
    all_imgs = []
    for cat_dir in sorted(RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            for img_path in cat_dir.glob(ext):
                try:
                    img = Image.open(img_path).convert('RGB')
                    h = imagehash.phash(img)
                    all_imgs.append((img_path, h, cat_dir.name))
                except Exception as e:
                    print(f"  SKIP {img_path.name}: {e}")
    
    print(f"  Total: {len(all_imgs)} images across categories")
    
    # Phase 2: Find duplicate groups
    print("[2] Finding duplicate groups...")
    groups = []  # list of sets of indices
    seen = set()
    
    for i in range(len(all_imgs)):
        if i in seen:
            continue
        group = {i}
        seen.add(i)
        hi = all_imgs[i][1]
        for j in range(i+1, len(all_imgs)):
            if j in seen:
                continue
            hj = all_imgs[j][1]
            if hi - hj <= hamming_threshold:
                group.add(j)
                seen.add(j)
        if len(group) > 1:
            groups.append(group)
    
    print(f"  Duplicate groups found: {len(groups)}")
    
    # Phase 3: For each group, keep first, mark others for deletion
    removed = []
    kept = []
    for group in groups:
        members = sorted([all_imgs[i] for i in group], key=lambda x: x[0].stat().st_size, reverse=True)
        # Keep the largest file (likely best quality), rest are duplicates
        keeper = members[0]
        kept.append(keeper[0])
        for dup in members[1:]:
            removed.append(dup[0])
    
    # Phase 4: Delete duplicates
    print(f"\n[3] Removing {len(removed)} duplicates...")
    per_cat_removed = defaultdict(int)
    for p in removed:
        cat = p.parent.name
        per_cat_removed[cat] += 1
        p.unlink()
    
    # Phase 5: Report
    print(f"\n[4] Summary:")
    print(f"  Kept:    {len(kept)} images (from duplicate groups)")
    print(f"  Removed: {len(removed)} duplicates")
    
    # Show per-category removal
    print(f"\n  Per-category removals:")
    for cat in sorted(per_cat_removed.keys()):
        remaining = len(list(Path(RAW_DIR / cat).glob("*.jpg"))) + \
                    len(list(Path(RAW_DIR / cat).glob("*.jpeg"))) + \
                    len(list(Path(RAW_DIR / cat).glob("*.png")))
        print(f"    {cat:<25s}: removed {per_cat_removed[cat]:>3d}, remaining ~{remaining}")
    
    # Show example groups
    print(f"\n[5] Example duplicate groups:")
    for group in groups[:5]:
        members = [all_imgs[i] for i in group]
        cats = set(m[2] for m in members)
        print(f"  Image appears in {len(members)} categories:")
        for m in members:
            print(f"    {m[2]:25s} -> {m[0].name}")
    
    # Phase 6: Count remaining
    total_remaining = 0
    for cat_dir in sorted(RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        count = len(list(cat_dir.glob("*.jpg"))) + \
                len(list(cat_dir.glob("*.jpeg"))) + \
                len(list(cat_dir.glob("*.png")))
        total_remaining += count
    
    print(f"\n[6] After dedup: {total_remaining} unique images")
    return total_remaining, len(removed)

if __name__ == "__main__":
    cross_dedup()

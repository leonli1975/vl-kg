"""Auto-generate blank annotation templates for all images."""

import json, os, shutil
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
ANNO_DIR = Path(__file__).parent.parent / "data" / "annotations"
TEMPLATE_PATH = ANNO_DIR / "template_blank.json"

def init_annotations():
    with open(TEMPLATE_PATH) as f:
        template = json.load(f)
    
    generated = 0
    for craft_dir in IMAGES_DIR.iterdir():
        if not craft_dir.is_dir():
            continue
        craft_type = "蜀锦" if craft_dir.name == "shujin" else "蜀绣"
        
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
            for img_path in craft_dir.glob(ext):
                anno_id = img_path.stem  # e.g., SJ_001
                anno_path = ANNO_DIR / f"{anno_id}.json"
                
                if anno_path.exists():
                    continue  # skip existing
                
                # Copy template and fill basics
                anno = json.loads(json.dumps(template))  # deep copy
                anno["image_id"] = anno_id
                anno["filename"] = img_path.name
                anno["craft"]["type"] = craft_type
                
                with open(anno_path, 'w', encoding='utf-8') as f:
                    json.dump(anno, f, indent=2, ensure_ascii=False)
                
                generated += 1
                print(f"  Created: {anno_path.name}")
    
    print(f"\nGenerated {generated} annotation templates.")
    print(f"Total annotations: {len(list(ANNO_DIR.glob('*.json')))}")
    
    # Show unfilled count
    unfilled = 0
    for p in ANNO_DIR.glob("*.json"):
        with open(p) as f:
            a = json.load(f)
        if not a.get("pattern_elements", [{}])[0].get("name"):
            unfilled += 1
    print(f"Unfilled: {unfilled}")

if __name__ == "__main__":
    init_annotations()
